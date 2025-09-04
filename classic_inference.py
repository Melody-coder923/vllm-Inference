import torch
from tokenizers import Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM


# 经典的完整推理流程
def classic_inference_step_by_step():
    # 1. 加载模型和分词器
    model_name = "gpt2"  #指定模型名称
    tokenizer = AutoTokenizer.from_pretrained(model_name)  #创建实例,加载分词器, 所谓加载即包含训练好的权重
    model = AutoModelForCausalLM.from_pretrained(model_name) # 加载模型, 所谓加载即包含完整词汇表
    model.eval() #设置模型为推理模式（关闭训练特性）
"""
eval()作用:
1. 关闭dropout（训练时随机丢弃神经元）
2. 关闭batch normalization的更新
3. 确保推理结果的一致性
"""
"""
from_pretrained() = 创建实例 + 加载预训练资源 + 配置参数
加载时的额外配置
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir="./my_cache",          # 自定义缓存目录
    use_fast=True,                   # 使用快速tokenizer
    trust_remote_code=True           # 信任远程代码（某些模型需要）
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir="./my_cache",          # 自定义缓存目录
    torch_dtype=torch.float16,      # 使用半精度节省内存
    device_map="auto",               # 自动分配GPU
    low_cpu_mem_usage=True          # 低CPU内存使用
)
"""
    # 2. 文本预处理（分词）
    prompt = "The future of AI is"
    # 编码（文本 → token IDs)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
"""
区分:
方法1: tokenize() - 只分词,  调试 - 看分词结果
tokens = tokenizer.tokenize(text)
print("tokenize():", tokens)
输出: ['Hello', ' world', '!']

方法2: encode() - 分词 + 转ID + 特殊token , 模型推理 - 🔥 最常用
token_ids = tokenizer.encode(text)
print("encode():", token_ids)
输出: [15496, 995, 0] (包含结束符)

可以互相转换
text = "Hello world"

路径1: 直接encode
token_ids = tokenizer.encode(text)

路径2: 先tokenize再convert
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
注意：这样不会自动添加特殊token！

反向转换
tokens = tokenizer.convert_ids_to_tokens(token_ids)
text = tokenizer.convert_tokens_to_string(tokens)
或者直接: text = tokenizer.decode(token_ids)
"""

    # 3. 逐步生成（autoregressive generation）
    max_new_tokens = 10
"""
含义：最多生成10个新的token
如何思考这个设置的考虑因素：
1. 任务需求 - 你想要多长的回答？
2. 计算成本 - 生成越多越慢
3. 质量控制 - 生成太长容易偏
max_new_tokens = 5   # 短句："bright and promising"
max_new_tokens = 20  # 中等："bright and will revolutionize many industries"
max_new_tokens = 100 # 长篇：完整段落
"""

    temperature = 0.8

"""
含义：控制生成的"创造性"程度
模型输出概率分布
原始概率 = [0.4, 0.3, 0.2, 0.1]  对应不同token的概率

temperature调整后：
temperature = 0.1 (保守)
调整概率 = [0.8, 0.15, 0.04, 0.01]  更确定性，选最可能的

temperature = 0.8 (平衡)
调整概率 = [0.45, 0.28, 0.18, 0.09]  # 适度随机

temperature = 2.0 (创造性)
调整概率 = [0.32, 0.28, 0.24, 0.16]  # 更随机，更有创意

完整的生成参数
generation_config = {
    "max_new_tokens": 10,
    "temperature": 0.8,
    "top_p": 0.9,          # 核采样，保留概率累积90%的token
    "top_k": 40,           # 只考虑概率最高的40个token
    "do_sample": True,     # 启用采样（而非贪婪解码）
    "repetition_penalty": 1.1,  # 避免重复
}
"""

    #在任何需要修改张量但又要保留原始数据的场景下，都应该使用 clone()！这是PyTorch编程的基本最佳实践。
    generated_ids = input_ids.clone() # 创建独立副本
    generation_time = time.time() #记录当前时间戳，用于性能测量！

    with torch.no_grad():
        for step in range(max_new_tokens):
            #print(f"\n--- Step {step + 1} ---")

            # 3a. 前向传播
            outputs = model(generated_ids)
"""
model() 不是内置方法 - 是Python的__call__魔法方法,实际执行 - 调用模型的forward方法
- 自动执行各种hook")
- 自动处理训练/评估模式"
不要调用forward()  跳过了hook机制",可能导致意外行为"

"""
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
"""
logits属于: outputs对象（返回值）
访问: outputs.logits
形状: [batch_size, seq_len, vocab_size]
[batch_size, sequence_length, vocab_size]"
批次大小 (一次处理几个句子); 序列长度 (输入有几个token); 词汇表大小 (GPT-2有50,257个词)
[    1     ,       4        ,   50257   ]
数值含义 logits[0, 3, 1234] = 5.67
[0]: 第1个样本; [3]: 第4个位置; [1234]: 词汇ID为1234的词;  5.67: 该词在该位置的'得分'
使用方式:获取最后一个位置的logits (用于预测下一个词)
last_logits = logits[0, -1, :]  # shape: [vocab_size]"
"""

    # 3b. 取最后一个位置的 logits
            next_token_logits = logits[0, -1, :]  # [vocab_size]
    #next_token_logits = logits[0, -1, :] 打印出来是 1维张量

    # 3c. 应用温度采样
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

    # 3d. 转换为概率分布, dim=-1 的优势：不管张量是几维，都指向最后一维
            probs = torch.softmax(next_token_logits, dim=-1)
"""
# dim=-1 的优势：不管张量是几维，都指向最后一维
tensor_1d = torch.randn(1000)          # [vocab_size]
tensor_2d = torch.randn(5, 1000)       # [batch, vocab_size]
tensor_3d = torch.randn(2, 5, 1000)    # [batch, seq, vocab_size]

# 都用 dim=-1，自动适应不同情况
probs_1d = torch.softmax(tensor_1d, dim=-1)  # 在维度0上
probs_2d = torch.softmax(tensor_2d, dim=-1)  # 在维度1上
probs_3d = torch.softmax(tensor_3d, dim=-1)  # 在维度2上
"""

    # 3e. 采样下一个 token
            next_token_id = torch.multinomial(probs, num_samples=1) #这里！根据概率分布随机选择

"""
def deeper_meaning():
采样的深层意义:

模拟不确定性:
真实世界充满不确定性
人类行为不是100%可预测的
采样让AI更接近人类的'随机性'

创造性来源:
创新往往来自'意外'选择"
低概率选项有时带来惊喜"
采样为AI提供了'灵感'的可能性"

避免局部最优:
贪婪选择容易陷入重复模式
采样提供了'跳出'的机会
让生成过程更加灵活

平衡性:
既不完全随机 (噪音)
也不完全确定 (无聊)"
在'合理'与'惊喜'间找平衡"

deeper_meaning()
"""


"""
采样的参数选择
def sampling_parameters():
影响采样的参数：
Temperature (温度):
低温 (0.1): 采样更倾向高概率词汇
中温 (0.8): 平衡的采样
高温 (2.0): 更多随机性，更多'冒险'选择"


Top-k 采样:
只在前k个最高概率词汇中采样
避免选择过于不合理的词汇


Top-p (Nucleus) 采样:
累积概率达到p时截止"
动态调整候选词汇数量


实际使用:
创意写作: 高temperature + top-p
技术文档: 低temperature + top-k
日常对话: 中等temperature + multinomial

"""
    # 3f. 添加到序列中
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=-1) #把刚生成的新词加到已有句子的末尾
"""
generated_ids：形状 [1, seq_len] - [batch_size, sequence_length]
next_token_id：形状 [1] - 需要变成 [1, 1]

需要匹配的维度结构：
第0维：batch_size（批次大小）
第1维：sequence_length（序列长度）

所以我们需要给 next_token_id 添加 batch 维度。

为什么需要 unsqueeze(0)？
这里有个维度匹配的问题：
next_token_id的形状：torch.multinomial()返回的是[1]，一个1维张量
generated_ids的形状：[1, seq_len]，一个2维张量（批次维度 × 序列维度）
直接拼接会出错，因为维度不匹配。unsqueeze(0)的作用是给next_token_id增加一个维度：

原来：[token_id] → 形状[1]
处理后：[[token_id]] → 形状[1, 1]

unsqueeze() 参数的含义
unsqueeze(dim) 会在指定位置 dim 插入一个大小为1的新维度。
具体例子
假设 next_token_id 的形状是 [1]：
unsqueeze(0) - 在第0个位置插入：

原来：[token_id] → 形状 [1]
结果：[[token_id]] → 形状 [1, 1]
理解：在最前面加了一个维度

unsqueeze(1) - 在第1个位置插入：

原来：[token_id] → 形状 [1]
结果：[[token_id]] → 形状 [1, 1]
理解：在后面加了一个维度

等等，两个结果一样？这是因为原张量只有1维，所以 unsqueeze(0) 和 unsqueeze(1) 效果相同。

记住这个规律：
unsqueeze(0) = "在前面加维度" = "加批次维度"
unsqueeze(-1) = "在后面加维度" = "加特征维度"
"""

"""
拼接前：
generated_ids:     [[15496, 995, 318, 389]]  形状：[1, 4]
next_token_id:     [[1049]]                   形状：[1, 1]

拼接操作：在dim=-1（最后一维）上连接
         ↓
结果：    [[15496, 995, 318, 389, 1049]]    形状：[1, 5]

"""

"""
一维张量拼接
[1, 2, 3] + [4] = [1, 2, 3, 4]
#   ↑                     ↑
# 原序列               末尾添加

# 只能用dim=0，因为只有这一个维度

二维张量拼接

[[1, 2, 3]] + [[4]] = [[1, 2, 3, 4]]
#    ↑                        ↑
#  原序列                  末尾添加

# dim=0: 会变成 [[1,2,3], [4]] ← 错误：创建新行
# dim=-1: 会变成 [[1,2,3,4]]  ← 正确：延长序列
"""

"""
torch.cat() 的工作原理
torch.cat()是PyTorch的张量拼接函数：

第一个参数：要拼接的张量列表,要拼接的多个张量. 要求：除了拼接维度外，其他维度大小必须相同.
第二个参数: 指定在哪个维度上进行拼接  dim=-1：在最后一个维度上拼接（也就是序列长度维度）
"""

    # 3g. 显示当前生成的文本
            current_text = tokenizer.decode(generated_ids[0])
            print(f"Current text: {current_text}")

    # 检查是否遇到停止条件
            if next_token_id.item() == tokenizer.eos_token_id:
                break
    generation_time = time.time() - generation_time #呼应前面,计算时间结束
    final_text = tokenizer.decode(generated_ids[0])
    return final_text
