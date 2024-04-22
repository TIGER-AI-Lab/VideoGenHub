from src.videogen_hub.pipelines.cogvideo.cogvideo_src.cogvideo_pipeline import (
    InferenceModel_Interpolate,
    InferenceModel_Sequential,
    my_filling_sequence,
    get_masks_and_position_ids_stage1,
    get_masks_and_position_ids_stage2,
    my_save_multiple_images,
)
from icetk import icetk as tokenizer
from src.videogen_hub.pipelines.cogvideo.cogvideo_src.coglm_strategy import (
    CoglmStrategy,
)
from src.videogen_hub.pipelines.cogvideo.cogvideo_src.sr_pipeline import (
    DirectSuperResolution,
)
from SwissArmyTransformer.resources import auto_create
import time, logging, sys, os, torch
import torch.distributed as dist

# path = os.path.join(args.output_path, f"{now_qi}_{raw_text}")


def pipeline(args, raw_text, height, width, duration):
    # model_stage1, args = InferenceModel_Sequential.from_pretrained(args, 'cogvideo-stage1')
    # model_stage1.eval()
    # parent_givan_tokens = process_stage1(model_stage1, raw_text, duration=4.0, video_raw_text=raw_text, video_guidance_text="视频",
    #                                         image_text_suffix=" 高清摄影",
    #                                         outputdir=None, batch_size=args.batch_size)

    # process_stage2(model_stage2, raw_text, duration=2.0, video_raw_text=raw_text+" 视频",
    #         video_guidance_text="视频", parent_given_tokens=parent_given_tokens,
    #         outputdir=path,
    #         gpu_rank=0, gpu_parallel_size=1) # TODO: 修改

    assert int(args.stage_1) + int(args.stage_2) + int(args.both_stages) == 1
    rank_id = args.device % args.parallel_size
    generate_frame_num = args.generate_frame_num

    if args.stage_1 or args.both_stages:
        model_stage1, args = InferenceModel_Sequential.from_pretrained(
            args, "cogvideo-stage1"
        )
        model_stage1.eval()
        if args.both_stages:
            model_stage1 = model_stage1.cpu()

    if args.stage_2 or args.both_stages:
        model_stage2, args = InferenceModel_Interpolate.from_pretrained(
            args, "cogvideo-stage2"
        )
        model_stage2.eval()
        if args.both_stages:
            model_stage2 = model_stage2.cpu()

    invalid_slices = [slice(tokenizer.num_image_tokens, None)]
    strategy_cogview2 = CoglmStrategy(invalid_slices, temperature=1.0, top_k=16)
    strategy_cogvideo = CoglmStrategy(
        invalid_slices,
        temperature=args.temperature,
        top_k=args.top_k,
        temperature2=args.coglm_temperature2,
    )
    if not args.stage_1:
        # from sr_pipeline import DirectSuperResolution
        dsr_path = auto_create(
            "cogview2-dsr", path=None
        )  # path=os.getenv('SAT_HOME', '~/.sat_models')
        dsr = DirectSuperResolution(args, dsr_path, max_bz=12, onCUDA=False)

    def process_stage2(
        model,
        seq_text,
        duration,
        video_raw_text=None,
        video_guidance_text="视频",
        parent_given_tokens=None,
        conddir=None,
        outputdir=None,
        gpu_rank=0,
        gpu_parallel_size=1,
    ):
        stage2_starttime = time.time()
        use_guidance = args.use_guidance_stage2
        if args.both_stages:
            move_start_time = time.time()
            logging.debug("moving stage-2 model to cuda")
            model = model.cuda()
            logging.debug(
                "moving in stage-2 model takes time: {:.2f}".format(
                    time.time() - move_start_time
                )
            )

        try:
            if parent_given_tokens is None:
                assert conddir is not None
                parent_given_tokens = torch.load(
                    os.path.join(conddir, "frame_tokens.pt"), map_location="cpu"
                )
            sample_num_allgpu = parent_given_tokens.shape[0]
            sample_num = sample_num_allgpu // gpu_parallel_size
            assert sample_num * gpu_parallel_size == sample_num_allgpu
            parent_given_tokens = parent_given_tokens[
                gpu_rank * sample_num : (gpu_rank + 1) * sample_num
            ]
        except:
            logging.critical("No frame_tokens found in interpolation, skip")
            return False

        # CogVideo Stage2 Generation
        while (
            duration >= 0.5
        ):  # TODO: You can change the boundary to change the frame rate
            parent_given_tokens_num = parent_given_tokens.shape[1]
            generate_batchsize_persample = (parent_given_tokens_num - 1) // 2
            generate_batchsize_total = generate_batchsize_persample * sample_num
            total_frames = generate_frame_num
            frame_len = 400
            enc_text = tokenizer.encode(seq_text)
            enc_duration = tokenizer.encode(str(float(duration)) + "秒")
            seq = (
                enc_duration
                + [tokenizer["<n>"]]
                + enc_text
                + [tokenizer["<start_of_image>"]]
                + [-1] * 400 * generate_frame_num
            )
            text_len = len(seq) - frame_len * generate_frame_num - 1

            logging.info(
                "[Stage2: Generating Frames, Frame Rate {:d}]\nraw text: {:s}".format(
                    int(4 / duration), tokenizer.decode(enc_text)
                )
            )

            # generation
            seq = (
                torch.cuda.LongTensor(seq, device=args.device)
                .unsqueeze(0)
                .repeat(generate_batchsize_total, 1)
            )
            for sample_i in range(sample_num):
                for i in range(generate_batchsize_persample):
                    seq[sample_i * generate_batchsize_persample + i][
                        text_len + 1 : text_len + 1 + 400
                    ] = parent_given_tokens[sample_i][2 * i]
                    seq[sample_i * generate_batchsize_persample + i][
                        text_len + 1 + 400 : text_len + 1 + 800
                    ] = parent_given_tokens[sample_i][2 * i + 1]
                    seq[sample_i * generate_batchsize_persample + i][
                        text_len + 1 + 800 : text_len + 1 + 1200
                    ] = parent_given_tokens[sample_i][2 * i + 2]

            if use_guidance:
                guider_seq = (
                    enc_duration
                    + [tokenizer["<n>"]]
                    + tokenizer.encode(video_guidance_text)
                    + [tokenizer["<start_of_image>"]]
                    + [-1] * 400 * generate_frame_num
                )
                guider_text_len = len(guider_seq) - frame_len * generate_frame_num - 1
                guider_seq = (
                    torch.cuda.LongTensor(guider_seq, device=args.device)
                    .unsqueeze(0)
                    .repeat(generate_batchsize_total, 1)
                )
                for sample_i in range(sample_num):
                    for i in range(generate_batchsize_persample):
                        guider_seq[sample_i * generate_batchsize_persample + i][
                            text_len + 1 : text_len + 1 + 400
                        ] = parent_given_tokens[sample_i][2 * i]
                        guider_seq[sample_i * generate_batchsize_persample + i][
                            text_len + 1 + 400 : text_len + 1 + 800
                        ] = parent_given_tokens[sample_i][2 * i + 1]
                        guider_seq[sample_i * generate_batchsize_persample + i][
                            text_len + 1 + 800 : text_len + 1 + 1200
                        ] = parent_given_tokens[sample_i][2 * i + 2]
                video_log_text_attention_weights = 0
            else:
                guider_seq = None
                guider_text_len = 0
                video_log_text_attention_weights = 1.4

            mbz = args.max_inference_batch_size

            assert generate_batchsize_total < mbz or generate_batchsize_total % mbz == 0
            output_list = []
            start_time = time.time()
            for tim in range(max(generate_batchsize_total // mbz, 1)):
                input_seq = (
                    seq[: min(generate_batchsize_total, mbz)].clone()
                    if tim == 0
                    else seq[mbz * tim : mbz * (tim + 1)].clone()
                )
                guider_seq2 = (
                    (
                        guider_seq[: min(generate_batchsize_total, mbz)].clone()
                        if tim == 0
                        else guider_seq[mbz * tim : mbz * (tim + 1)].clone()
                    )
                    if guider_seq is not None
                    else None
                )
                output_list.append(
                    my_filling_sequence(
                        model,
                        args,
                        input_seq,
                        batch_size=min(generate_batchsize_total, mbz),
                        get_masks_and_position_ids=get_masks_and_position_ids_stage2,
                        text_len=text_len,
                        frame_len=frame_len,
                        strategy=strategy_cogview2,
                        strategy2=strategy_cogvideo,
                        log_text_attention_weights=video_log_text_attention_weights,
                        mode_stage1=False,
                        guider_seq=guider_seq2,
                        guider_text_len=guider_text_len,
                        guidance_alpha=args.guidance_alpha,
                        limited_spatial_channel_mem=True,
                    )[0]
                )
            logging.info(
                "Duration {:.2f}, Taken time {:.2f}\n".format(
                    duration, time.time() - start_time
                )
            )

            output_tokens = torch.cat(output_list, dim=0)
            output_tokens = output_tokens[
                :, text_len + 1 : text_len + 1 + (total_frames) * 400
            ].reshape(sample_num, -1, 400 * total_frames)
            output_tokens_merge = torch.cat(
                (
                    output_tokens[:, :, : 1 * 400],
                    output_tokens[:, :, 400 * 3 : 4 * 400],
                    output_tokens[:, :, 400 * 1 : 2 * 400],
                    output_tokens[:, :, 400 * 4 : (total_frames) * 400],
                ),
                dim=2,
            ).reshape(sample_num, -1, 400)

            output_tokens_merge = torch.cat(
                (output_tokens_merge, output_tokens[:, -1:, 400 * 2 : 3 * 400]), dim=1
            )
            duration /= 2
            parent_given_tokens = output_tokens_merge

        if args.both_stages:
            move_start_time = time.time()
            logging.debug("moving stage 2 model to cpu")
            model = model.cpu()
            torch.cuda.empty_cache()
            logging.debug(
                "moving out model2 takes time: {:.2f}".format(
                    time.time() - move_start_time
                )
            )

        logging.info(
            "CogVideo Stage2 completed. Taken time {:.2f}\n".format(
                time.time() - stage2_starttime
            )
        )

        # decoding
        # imgs = [torch.nn.functional.interpolate(tokenizer.decode(image_ids=seq.tolist()), size=(480, 480)) for seq in output_tokens_merge]
        # os.makedirs(output_dir_full_path, exist_ok=True)
        # my_save_multiple_images(imgs, output_dir_full_path,subdir="frames", debug=False)
        # torch.save(output_tokens_merge.cpu(), os.path.join(output_dir_full_path, 'frame_token.pt'))
        # os.system(f"gifmaker -i '{output_dir_full_path}'/frames/0*.jpg -o '{output_dir_full_path}/{str(float(duration))}_concat.gif' -d 0.2")

        # direct super-resolution by CogView2
        logging.info("[Direct super-resolution]")
        dsr_starttime = time.time()
        enc_text = tokenizer.encode(seq_text)
        frame_num_per_sample = parent_given_tokens.shape[1]
        parent_given_tokens_2d = parent_given_tokens.reshape(-1, 400)
        text_seq = (
            torch.cuda.LongTensor(enc_text, device=args.device)
            .unsqueeze(0)
            .repeat(parent_given_tokens_2d.shape[0], 1)
        )
        sred_tokens = dsr(text_seq, parent_given_tokens_2d)
        decoded_sr_videos = []

        for sample_i in range(sample_num):
            decoded_sr_imgs = []
            for frame_i in range(frame_num_per_sample):
                decoded_sr_img = tokenizer.decode(
                    image_ids=sred_tokens[frame_i + sample_i * frame_num_per_sample][
                        -3600:
                    ]
                )
                decoded_sr_imgs.append(
                    torch.nn.functional.interpolate(
                        decoded_sr_img, size=(height, width)
                    )
                )
            decoded_sr_videos.append(decoded_sr_imgs)

        return decoded_sr_videos
        # for sample_i in range(sample_num):
        #     my_save_multiple_images(decoded_sr_videos[sample_i], outputdir,subdir=f"frames/{sample_i+sample_num*gpu_rank}", debug=False)
        #     os.system(f"gifmaker -i '{outputdir}'/frames/'{sample_i+sample_num*gpu_rank}'/0*.jpg -o '{outputdir}/{sample_i+sample_num*gpu_rank}.gif' -d 0.125")

        # logging.info("Direct super-resolution completed. Taken time {:.2f}\n".format(time.time() - dsr_starttime))

        # return True

    def process_stage1(
        model,
        seq_text,
        duration,
        video_raw_text=None,
        video_guidance_text="视频",
        image_text_suffix="",
        outputdir=None,
        batch_size=1,
    ):
        process_start_time = time.time()
        use_guide = args.use_guidance_stage1
        if args.both_stages:
            move_start_time = time.time()
            logging.debug("moving stage 1 model to cuda")
            model = model.cuda()
            logging.debug(
                "moving in model1 takes time: {:.2f}".format(
                    time.time() - move_start_time
                )
            )

        if video_raw_text is None:
            video_raw_text = seq_text
        mbz = (
            args.stage1_max_inference_batch_size
            if args.stage1_max_inference_batch_size > 0
            else args.max_inference_batch_size
        )
        assert batch_size < mbz or batch_size % mbz == 0
        frame_len = 400

        # generate the first frame:
        enc_text = tokenizer.encode(seq_text + image_text_suffix)
        seq_1st = (
            enc_text + [tokenizer["<start_of_image>"]] + [-1] * 400
        )  # IV!!  # test local!!! # test randboi!!!
        logging.info(
            "[Generating First Frame with CogView2]Raw text: {:s}".format(
                tokenizer.decode(enc_text)
            )
        )
        text_len_1st = len(seq_1st) - frame_len * 1 - 1

        seq_1st = torch.cuda.LongTensor(seq_1st, device=args.device).unsqueeze(0)
        output_list_1st = []
        for tim in range(max(batch_size // mbz, 1)):
            start_time = time.time()
            output_list_1st.append(
                my_filling_sequence(
                    model,
                    args,
                    seq_1st.clone(),
                    batch_size=min(batch_size, mbz),
                    get_masks_and_position_ids=get_masks_and_position_ids_stage1,
                    text_len=text_len_1st,
                    frame_len=frame_len,
                    strategy=strategy_cogview2,
                    strategy2=strategy_cogvideo,
                    log_text_attention_weights=1.4,
                    enforce_no_swin=True,
                    mode_stage1=True,
                )[0]
            )
            logging.info(
                "[First Frame]Taken time {:.2f}\n".format(time.time() - start_time)
            )
        output_tokens_1st = torch.cat(output_list_1st, dim=0)
        given_tokens = output_tokens_1st[
            :, text_len_1st + 1 : text_len_1st + 401
        ].unsqueeze(
            1
        )  # given_tokens.shape: [bs, frame_num, 400]

        # generate subsequent frames:
        total_frames = generate_frame_num
        enc_duration = tokenizer.encode(str(float(duration)) + "秒")
        if use_guide:
            video_raw_text = video_raw_text + " 视频"
        enc_text_video = tokenizer.encode(video_raw_text)
        seq = (
            enc_duration
            + [tokenizer["<n>"]]
            + enc_text_video
            + [tokenizer["<start_of_image>"]]
            + [-1] * 400 * generate_frame_num
        )
        guider_seq = (
            enc_duration
            + [tokenizer["<n>"]]
            + tokenizer.encode(video_guidance_text)
            + [tokenizer["<start_of_image>"]]
            + [-1] * 400 * generate_frame_num
        )
        logging.info(
            "[Stage1: Generating Subsequent Frames, Frame Rate {:.1f}]\nraw text: {:s}".format(
                4 / duration, tokenizer.decode(enc_text_video)
            )
        )

        text_len = len(seq) - frame_len * generate_frame_num - 1
        guider_text_len = len(guider_seq) - frame_len * generate_frame_num - 1
        seq = (
            torch.cuda.LongTensor(seq, device=args.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        guider_seq = (
            torch.cuda.LongTensor(guider_seq, device=args.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

        for given_frame_id in range(given_tokens.shape[1]):
            seq[
                :,
                text_len
                + 1
                + given_frame_id * 400 : text_len
                + 1
                + (given_frame_id + 1) * 400,
            ] = given_tokens[:, given_frame_id]
            guider_seq[
                :,
                guider_text_len
                + 1
                + given_frame_id * 400 : guider_text_len
                + 1
                + (given_frame_id + 1) * 400,
            ] = given_tokens[:, given_frame_id]
        output_list = []

        if use_guide:
            video_log_text_attention_weights = 0
        else:
            guider_seq = None
            video_log_text_attention_weights = 1.4

        for tim in range(max(batch_size // mbz, 1)):
            start_time = time.time()
            input_seq = (
                seq[: min(batch_size, mbz)].clone()
                if tim == 0
                else seq[mbz * tim : mbz * (tim + 1)].clone()
            )
            guider_seq2 = (
                (
                    guider_seq[: min(batch_size, mbz)].clone()
                    if tim == 0
                    else guider_seq[mbz * tim : mbz * (tim + 1)].clone()
                )
                if guider_seq is not None
                else None
            )
            output_list.append(
                my_filling_sequence(
                    model,
                    args,
                    input_seq,
                    batch_size=min(batch_size, mbz),
                    get_masks_and_position_ids=get_masks_and_position_ids_stage1,
                    text_len=text_len,
                    frame_len=frame_len,
                    strategy=strategy_cogview2,
                    strategy2=strategy_cogvideo,
                    log_text_attention_weights=video_log_text_attention_weights,
                    guider_seq=guider_seq2,
                    guider_text_len=guider_text_len,
                    guidance_alpha=args.guidance_alpha,
                    limited_spatial_channel_mem=True,
                    mode_stage1=True,
                )[0]
            )

        output_tokens = torch.cat(output_list, dim=0)[:, 1 + text_len :]

        if args.both_stages:
            move_start_time = time.time()
            logging.debug("moving stage 1 model to cpu")
            model = model.cpu()
            torch.cuda.empty_cache()
            logging.debug(
                "moving in model1 takes time: {:.2f}".format(
                    time.time() - move_start_time
                )
            )

        # decoding
        imgs, sred_imgs, txts = [], [], []
        for seq in output_tokens:
            decoded_imgs = [
                torch.nn.functional.interpolate(
                    tokenizer.decode(image_ids=seq.tolist()[i * 400 : (i + 1) * 400]),
                    size=(height, width),
                )
                for i in range(total_frames)
            ]
            imgs.append(decoded_imgs)  # only the last image (target)

        assert len(imgs) == batch_size
        return imgs
        # save_tokens = output_tokens[:, :+total_frames*400].reshape(-1, total_frames, 400).cpu()
        # if outputdir is not None:
        #     for clip_i in range(len(imgs)):
        #         # os.makedirs(output_dir_full_paths[clip_i], exist_ok=True)
        #         my_save_multiple_images(imgs[clip_i], outputdir, subdir=f"frames/{clip_i}", debug=False)
        #         os.system(f"gifmaker -i '{outputdir}'/frames/'{clip_i}'/0*.jpg -o '{outputdir}/{clip_i}.gif' -d 0.25")
        #     torch.save(save_tokens, os.path.join(outputdir, 'frame_tokens.pt'))

        # logging.info("CogVideo Stage1 completed. Taken time {:.2f}\n".format(time.time() - process_start_time))

        # return save_tokens

    # ======================================================================================================

    if args.stage_1 or args.both_stages:
        if args.input_source != "interactive":
            with open(args.input_source, "r") as fin:
                promptlist = fin.readlines()
            promptlist = [p.strip() for p in promptlist]
        else:
            promptlist = None

        now_qi = -1
        while True:
            now_qi += 1

            if promptlist is not None:  # with input-source
                if args.multi_gpu:
                    if now_qi % dist.get_world_size() != dist.get_rank():
                        continue
                    rk = dist.get_rank()
                else:
                    rk = 0
                raw_text = promptlist[now_qi]
                raw_text = raw_text.strip()
                print(f"Working on Line No. {now_qi} on {rk}... [{raw_text}]")
            else:  # interactive
                raw_text = input("\nPlease Input Query (stop to exit) >>> ")
                raw_text = raw_text.strip()
                if not raw_text:
                    print("Query should not be empty!")
                    continue
                if raw_text == "stop":
                    return

            try:
                path = os.path.join(args.output_path, f"{now_qi}_{raw_text}")
                parent_given_tokens, imgs = process_stage1(
                    model_stage1,
                    raw_text,
                    duration=4.0,
                    video_raw_text=raw_text,
                    video_guidance_text="视频",
                    image_text_suffix=" 高清摄影",
                    outputdir=path if args.stage_1 else None,
                    batch_size=args.batch_size,
                )
                if args.stage_1 and not args.both_stages:
                    print("only stage 1")
                    return imgs

                if args.both_stages:
                    videos = process_stage2(
                        model_stage2,
                        raw_text,
                        duration=duration,
                        video_raw_text=raw_text + " 视频",
                        video_guidance_text="视频",
                        parent_given_tokens=parent_given_tokens,
                        outputdir=path,
                        gpu_rank=0,
                        gpu_parallel_size=1,
                    )  # TODO: 修改
                    return videos
            except (ValueError, FileNotFoundError) as e:
                print(e)
                continue

    elif args.stage_2:
        sample_dirs = os.listdir(args.output_path)
        for sample in sample_dirs:
            raw_text = sample.split("_")[-1]
            path = os.path.join(args.output_path, sample, "Interp")
            parent_given_tokens = torch.load(
                os.path.join(args.output_path, sample, "frame_tokens.pt")
            )

            process_stage2(
                raw_text,
                duration=2.0,
                video_raw_text=raw_text + " 视频",
                video_guidance_text="视频",
                parent_given_tokens=parent_given_tokens,
                outputdir=path,
                gpu_rank=0,
                gpu_parallel_size=1,
            )  # TODO: 修改

    else:
        assert False
