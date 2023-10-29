import os
import cv2
import torch
from tqdm import tqdm
import warnings
import _thread
from queue import Queue
from models.model_union_2.RIFE import Model

warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)


n_forward = 2  # 用户可以指定, N - 1, 表示解决一拍N问题, 最小为1 (程序执行结束后会吃掉开头的N帧)
times = 5  # 补帧倍数 fps_in -> downfps = fps_in / n_forward -> (downfps * times)


video = r''  # 输入视频
save = r''  # 保存输出图片序列的路径
scale = 1.0  # 光流缩放尺度
global_size = (1920, 1088)  # 全局图像尺寸(自行pad)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
model = Model()
if not hasattr(model, 'version'):
    model.version = 0
model.load_model('train_logs/v', -1)
print("Loaded model")
model.eval()
model.device()


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


def decrase_inference(inputs: list, saved_result: dict, layers=0, counter=0):
    layers += 1
    if len(inputs) == 1:
        # return inputs[0], saved_result, counter  # 字典为可变序列, 不需要返回
        return inputs[0], counter
    tmp_queue = []
    for i in range(len(inputs) - 1):
        # 先读表, 不重复生成结果 (超大幅度加速计算)
        if saved_result.get(f'{layers}{i + 1}') is not None:
            saved_result[f'{layers}{i}'] = saved_result[f'{layers}{i + 1}']   # 向前移动整个倒三角 (可以忽略这行注释)
            tmp_queue.append(
                saved_result[f'{layers}{i}']
            )
        else:
            # 反复执行to_tensor -> to_numpy可以节省显存, 但可能会显著降低执行速度
            inp0, inp1 = map(to_tensor, [inputs[i], inputs[i + 1]])
            tmp_queue.append(
                to_numpy(model.inference(inp0, inp1, model.reuse(inp0, inp1, scale), 0.5, scale))
            )
            saved_result[f'{layers}{i}'] = tmp_queue[-1]   # 补充倒三角 (可以忽略这行注释)
            counter += 1
    return decrase_inference(tmp_queue, saved_result, layers, counter)


pbar.update(n_forward)  # 初始化进度

# 初始化输入序列
i0 = get()
# put(i0)?  # 是否处理片头缺失帧
queue_input = [i0]
queue_output = []
saved_result = {}
output0 = None

while True:
    if output0 is None:
        for i in range(n_forward):
            queue_input.append(get())
        # output0, saved_result, count = decrase_inference(queue_input.copy(), saved_result)  # 字典为可变序列, 不需要返回
        # 列表queue_input为可变序列, 使用copy避免改变
        output0, count = decrase_inference(queue_input.copy(), saved_result)
        # print(f"首次计算推理次数: {count}")

    _ = queue_input.pop(0)
    queue_input.append(get())
    if queue_input[-1] is None:
        break

    # output1, saved_result, count = decrase_inference(queue_input.copy(), saved_result)   # 字典为可变序列, 不需要返回
    # 列表queue_input为可变序列, 使用copy避免改变
    output1, count = decrase_inference(queue_input.copy(), saved_result)
    # print(f"前进计算推理次数: {count}")

    queue_output.append(output0)
    inp0, inp1 = map(to_tensor, [output0, output1])
    reuse_things = model.reuse(inp0, inp1, scale)
    for t in [(1 / (times - 1)) * i for i in range(1, times)]:
        out = to_numpy(model.inference(inp0, inp1, reuse_things, t, scale))
        queue_output.append(out)

    for out in queue_output:
        # 在queue_output中已经转换过输出
        # 在推理过程中反复执行to_tensor -> to_numpy可以节省显存, 但可能会显著降低执行速度
        put(out)

    queue_output.clear()
    output0 = output1
    pbar.update(1)

put(output0.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.)
pbar.update(1)
