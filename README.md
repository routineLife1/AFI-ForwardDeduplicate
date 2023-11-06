# AFI-ForwardDeduplicate
Efficient Deduplicate for Anime Video Frame Interpolation

![ezgif com-video-to-gif](https://github.com/hyw-dev/AFI-ForwardDeduplicate/assets/68835291/6f03dfd8-99f4-48ad-871e-91cbd704c1e5)

#  Demonstrations
### [bilibili](https://www.bilibili.com/video/BV1py4y1A7qj)


# After completing the todo list, usage instructions will be released.

# todo list
- [ ] ~~**Efficiency optimization**~~ (No significant efficiency gains and increased risk of vram overflow.)
- [ ] ~~**Attempt to implement arbitrary frame rates support**~~ (It is very inefficient to process all the frames in a scene at once, or to accurately calculate the total number of frames variety of videos.)
- [ ] ~~**Attempt to accurately determine transition even in the queue_input**~~ (The implementation code is too complex, and it's effect is not obvious to improve)
- [x] **Improve the smoothness By reducing transition frames to one frame and allocate them to the end of the scene**

# limitations and expectations
> 1. It is temporarily impossible to dynamically adjust the "n_forward" parameter through MAX animation rhythm.
> If it can be supported, we can get the smoothest result in one step, and it will definitely surpass manual deduplication.
>
> 2. the "n_forward" parameter acts like the number of times the algorithm performs TTA (Test Time Augmentation) operations.
> Performing too many TTA operations may lead to blurriness.

## Reference
[SpatiotemporalResampling](https://github.com/hyw-dev/SpatiotemporalResampling) [GMFSS](https://github.com/98mxr/GMFSS_Fortuna) [Practical-RIFE](https://github.com/hzwer/Practical-RIFE)
