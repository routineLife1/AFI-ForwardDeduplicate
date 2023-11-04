# AFI-ForwardDeduplicate
Efficient Deduplicate for Anime Video Frame Interpolation

![ezgif com-video-to-gif](https://github.com/hyw-dev/AFI-ForwardDeduplicate/assets/68835291/6f03dfd8-99f4-48ad-871e-91cbd704c1e5)

#  Demonstrations
### [bilibili](https://www.bilibili.com/video/BV1py4y1A7qj)


# After completing the todo list, usage instructions will be released.

# todo list
- [ ] **Efficiency optimization**
- [ ] **Attempt to implement arbitrary frame rates support**
- [ ] **Attempt to accurately determine transition even in the queue_input**
- [ ] **Reduce transition frames to one frame and allocate them to the end of the scene**

# limitations and expectations
> It is temporarily impossible to dynamically adjust the "n_forward" parameter through animation rhythm.
> If it can be supported, the effect will be further improved, and it will definitely surpass manual deduplication.
> The reason for not directly setting the maximum possible number of beats is because the "n_forward" parameter acts like the number of times the algorithm performs TTA (Test Time Augmentation) operations.
> Performing too many TTA operations may lead to blurriness and lag. The exact cause of lag caused by excessive TTA operations is currently unknown.

## Reference
[SpatiotemporalResampling](https://github.com/hyw-dev/SpatiotemporalResampling) [GMFSS](https://github.com/98mxr/GMFSS_Fortuna) [Practical-RIFE](https://github.com/hzwer/Practical-RIFE)
