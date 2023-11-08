# 📖AFI-ForwardDeduplicate
Efficient Deduplicate for Anime Video Frame Interpolation

![ezgif com-video-to-gif](https://github.com/hyw-dev/AFI-ForwardDeduplicate/assets/68835291/6f03dfd8-99f4-48ad-871e-91cbd704c1e5)

## 👀Demos Videos
### [bilibili](https://www.bilibili.com/video/BV1py4y1A7qj)

## 🔧Dependencies
- ffmpeg
- same as [GMFSS](https://github.com/98mxr/GMFSS_Fortuna)
- download the [weights](https://drive.google.com/file/d/157M4i1B9hjWs1K2AZVArSulkM9qV2sdH/view?usp=sharing) and unzip it, put them to ./weights/
- see issue [#2](https://github.com/hyw-dev/AFI-ForwardDeduplicate/issues/2) to solve ImportError
 
## ⚡Usage 
- normalize the source video to 24000/1001 fps by following command using ffmpeg
  ```bash
  ffmpeg -i INPUT -crf 16 -r 24000/1001 -preset slow -c:v libx265 -x265-params profile=main10 -c:a copy OUTPUT
  ```
- open the video and check out it's max consistent deduplication counts, (3 -> on Three, 2 -> on Two)
- run the follwing command to finish interpolation
  (N_FORWARD = max_consistent_deduplication_counts - 1)
  ```bash
  python interpolate_video_forward.py -i [VIDEO] -o [OUTPUTDIR] -nf [N_FORWARD] -t [TIMES] -m [MODEL_TYPE] -s [ENABLE_SCDET] -st 14 -stf True -scale [SCALE]
  ```
  
- run the follwing command or custom command to merge the output frames with the audio of source video
  ```bash
  ffmpeg -r [24000/1001 * TIMES] -i [OUTPUTDIR]/%09d.png -i [VIDEO] -map 0:v -map 1:a -crf 16 -preset slow -c:v libx265 -x265-params profile=main10 -c:a copy [FINAL_OUTPUT]
  ```
  
 **example(smooth a 23.976fps video with on three and interpolate it to 47.952fps):**

  ```bash
  ffmpeg -i E:/Myvideo/01_src.mkv -crf 16 -r 24000/1001 -preset slow -c:v libx265 -x265-params profile=main10 -c:a copy E:/Myvideo/01.mkv

  python interpolate_video_forward.py -i E:/MyVideo/01.mkv -o E:/frame_seq_output -nf 2 -t 2 -m gmfss -s True -st 14 -stf True -scale 1.0

  ffmpeg -r 47.952 -i E:/frame_seq_output/%09d.png -i E:/MyVideo/01.mkv -map 0:v -map 1:a -crf 16 -preset slow -c:v libx265 -x265-params profile=main10 -c:a copy E:/final_output/01.mkv
  ```
  

## todo list
- [ ] ~~**Efficiency optimization**~~ (No significant efficiency gains and increased risk of vram overflow.)
- [ ] ~~**Attempt to implement arbitrary frame rates support**~~ (It is very inefficient to process all the frames in a scene at once, or to accurately calculate the total number of frames variety of videos.)
- [ ] ~~**Attempt to accurately determine transition even in the queue_input**~~ (The implementation code is too complex, and it's effect is not obvious to improve)
- [x] **Improve the smoothness By reducing transition frames to one frame and allocate them to the end of the scene**
- [ ] **Explain why this method is effective and write a guidence on how to support other vfi algorithms**

## limitations and expectations
> 1. It is temporarily impossible to dynamically adjust the "n_forward" parameter through auto detect the max consistent deduplication counts.
> If it can be supported, we can get the smoothest result in one step, and it will definitely surpass manual deduplication.
>
> 2. the "n_forward" parameter acts like the number of times the algorithm performs TTA (Test Time Augmentation) operations.
> Performing too many TTA operations may lead to blurriness.

## Projects that use AFI-ForwardDeduplicate
[SVFI(commercial software)](https://store.steampowered.com/app/1692080/SVFI/)

## Reference
[SpatiotemporalResampling](https://github.com/hyw-dev/SpatiotemporalResampling) [GMFSS](https://github.com/98mxr/GMFSS_Fortuna) [Practical-RIFE](https://github.com/hzwer/Practical-RIFE)
