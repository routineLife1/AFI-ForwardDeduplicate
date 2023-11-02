import os
import cv2
import torch
from tqdm import tqdm
import warnings
import _thread
import time
from queue import Queue
from models.model_pg104.RIFE import Model
from IFNet_HDv3 import IFNet

warnings.warn(
    "正常情况下补帧操作无法得到frames * times帧, 正常应该得到times * (frames - 1) + 1帧, 请在程序结束后检查总帧数")
warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)

model_type = 'gmfss'  # gmfss / rife
n_forward = 2  # 解决一拍N及以下问题, 则输入值N-1, 最小为1
times = 5  # 补帧倍数 >= 2

video = r'E:\Work\VFI\Algorithm\GMFwSS\test_recon\material\1.mp4'  # 输入视频
save = r'E:\Work\VFI\Algorithm\GMFwSS\output'  # 保存输出图片序列的路径
scale = 1.0  # 光流缩放尺度
global_size = (960, 576)  # 全局图像尺寸(自行pad)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def convert(param):
    return {
        k.replace("module.", ""): v
        for k, v in param.items()
        if "module." in k
    }


if model_type == 'rife':
    model = IFNet()
    model.load_state_dict(convert(torch.load('rife48.pkl')))
else:
    model = Model()
    model.load_model('train_logs/old/train_log_pg104', -1)
model.eval()
if model_type == 'gmfss':
    model.device()
else:
    model.to(device).half()
print("Loaded model")


def to_tensor(img):
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).half().cuda() / 255.


def to_numpy(tensor):
    return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.


output_counter = 0  # 输出计数器


def put(things):  # 将输出帧推送至write_buffer
    global output_counter
    output_counter += 1
    write_buffer.put([output_counter, things])


def get():  # 获取输入帧
    return read_buffer.get()


def build_read_buffer(r_buffer, v):
    ret, __x = v.read()
    while ret:
        r_buffer.put(cv2.resize(__x, global_size))
        ret, __x = v.read()
    r_buffer.put(None)


def clear_write_buffer(w_buffer):
    while True:
        item = w_buffer.get()
        if item is None:
            break
        num = item[0]
        content = item[1]
        cv2.imwrite(os.path.join(save, "{:0>9d}.png".format(num)), cv2.resize(content, global_size))


video_capture = cv2.VideoCapture(video)
total_frames_count = video_capture.get(7)
read_buffer = Queue(maxsize=100)
write_buffer = Queue(maxsize=-1)
_thread.start_new_thread(build_read_buffer, (read_buffer, video_capture))
_thread.start_new_thread(clear_write_buffer, (write_buffer,))
pbar = tqdm(total=total_frames_count - n_forward)


def make_inf(x, y, scale, timestep):
    if model_type == 'rife':
        return model(torch.cat((x, y), dim=1), timestep)
    return model.inference(x, y, model.reuse(x, y, scale), timestep)


def decrase_inference(inputs: list, saved_result: dict, layers=0, counter=0):
    while len(inputs) != 1:
        layers += 1
        tmp_queue = []
        for i in range(len(inputs) - 1):
            # 先读表, 不重复生成结果 (超大幅度加速计算)
            if saved_result.get(f'{layers}{i + 1}') is not None:
                saved_result[f'{layers}{i}'] = saved_result[
                    f'{layers}{i + 1}']  # 向前移动整个倒三角 (可以忽略这行注释), 这样存储会增加空间复杂度, 建议简化
                tmp_queue.append(
                    saved_result[f'{layers}{i}']
                )
            else:
                # 反复执行to_tensor -> to_numpy可以节省显存, 但可能会显著降低执行速度
                inp0, inp1 = map(to_tensor, [inputs[i], inputs[i + 1]])
                tmp_queue.append(
                    to_numpy(make_inf(inp0, inp1, scale, 0.5))
                )
                saved_result[f'{layers}{i}'] = tmp_queue[-1]  # 补充倒三角 (可以忽略这行注释), 这样存储会增加空间复杂度, 建议简化
                counter += 1
        inputs = tmp_queue
    # return inputs[0], saved_result, counter  # 字典为可变序列, 不需要返回
    return inputs[0], counter


def gen_ts_frame(x, y, _scale, ts):
    _outputs = list()
    _reuse_things = model.reuse(x, y, _scale) if model_type == 'gmfss' else None
    for t in ts:
        if model_type == 'rife':
            _out = make_inf(inp0, inp1, _scale, t)
        else:
            _out = model.inference(inp0, inp1, _reuse_things, t, _scale)
        _outputs.append(to_numpy(_out))
    return _outputs


# 初始化输入序列
i0 = get()  # 0
queue_input = [i0]
queue_output = []
saved_result = {}
output0 = None
# if times = 5, n_forward=2, right=4, left=4
right_infill = (times * n_forward) // 2 - 1
left_infill = right_infill + (times * n_forward) % 2
times_ts = [i / times for i in range(1, times)]

while True:
    if output0 is None:
        queue_input.extend(get() for _ in range(n_forward))  # 1, 2 (n_forward=2)
        # output0, saved_result, count = decrase_inference(queue_input.copy(), saved_result)  # 字典为可变序列, 不需要返回
        # 列表queue_input为可变序列, 使用copy避免改变

        output0, count = decrase_inference(queue_input.copy(), saved_result)  # 使用 0,1,2
        # print(f"首次计算推理次数: {count}")

        queue_output.append(i0)  # 开头帧, 规定必须放
        inputs = [i0]
        inputs.extend(saved_result[f'{layer}0'] for layer in range(1, n_forward + 1))

        timestamp = [0.5 * layer for layer in range(0, n_forward + 1)]  # np.linspace()
        t_step = timestamp[-1] / (left_infill + 1)
        require_timestamp = [t_step * i for i in range(1, left_infill + 1)]  # np.linspace()

        for i in range(len(timestamp) - 1):
            t0, t1 = timestamp[i], timestamp[i + 1]
            if t0 in require_timestamp:
                queue_output.append(inputs[i])
                require_timestamp.remove(t0)
            if t1 in require_timestamp:
                queue_output.append(inputs[i + 1])
                require_timestamp.remove(t1)
            condition_middle = [rt for rt in require_timestamp if t0 < rt < t1]
            if len(condition_middle) != 0:
                inp0, inp1 = map(to_tensor, [inputs[i], inputs[i + 1]])
                outputs = gen_ts_frame(inp0, inp1, scale, [(t - t0) * 2 for t in condition_middle])
                queue_output.extend(outputs)
            if len(require_timestamp) == 0:
                break

    # 向前推进
    _ = queue_input.pop(0)
    queue_input.append(get())  # 1, 2, 3

    # 读到帧尾
    if queue_input[-1] is None:
        queue_output = [output0]
        inputs = [output0]
        inputs.extend(saved_result[f'{layer}{n_forward - layer}'] for layer in range(n_forward - 1, 0, -1))
        inputs.append(queue_input[-2])

        timestamp = [0.5 * layer for layer in range(0, n_forward + 1)]  # np.linspace()
        t_step = timestamp[-1] / (right_infill + 1)
        require_timestamp = [t_step * i for i in range(1, right_infill + 1)]  # np.linspace()

        for i in range(len(timestamp) - 1):
            t0, t1 = timestamp[i], timestamp[i + 1]
            if t0 in require_timestamp:
                queue_output.append(inputs[i])
                require_timestamp.remove(t0)
            if t1 in require_timestamp:
                queue_output.append(inputs[i + 1])
                require_timestamp.remove(t1)
            condition_middle = [rt for rt in require_timestamp if t0 < rt < t1]
            if len(condition_middle) != 0:
                inp0, inp1 = map(to_tensor, [inputs[i], inputs[i + 1]])
                outputs = gen_ts_frame(inp0, inp1, scale, [(t - t0) * 2 for t in condition_middle])
                queue_output.extend(outputs)
            if len(require_timestamp) == 0:
                break
        queue_output.append(queue_input[-2])

        for out in queue_output:
            put(out)

        break

    # output1, saved_result, count = decrase_inference(queue_input.copy(), saved_result)   # 字典为可变序列, 不需要返回
    # 列表queue_input为可变序列, 使用copy避免改变
    output1, count = decrase_inference(queue_input.copy(), saved_result)  # 输入 1,2,3
    # print(f"前进计算推理次数: {count}")

    queue_output.append(output0)
    inp0, inp1 = map(to_tensor, [output0, output1])
    queue_output.extend(gen_ts_frame(inp0, inp1, scale, times_ts))

    for out in queue_output:
        # 在queue_output中已经转换过输出
        # 在推理过程中反复执行to_tensor -> to_numpy可以节省显存, 但可能会显著降低执行速度
        put(out)

    queue_output.clear()
    output0 = output1
    pbar.update(1)

# 等待帧全部导出完成
print('Wait for all frames to be exported...')
while not write_buffer.empty():
    time.sleep(0.1)

pbar.update(1)
print('Done!')
