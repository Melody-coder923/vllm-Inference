from vllm import LLM, SamplingParams


def vllm_inference():
    # vLLM 把所有步骤都封装了
    llm = LLM(model="gpt2")
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=10,
        stop_token_ids=[50256]  # GPT2 的 EOS token
    )

    generation_time = time.time()
    prompt = "The future of AI is"
    outputs = llm.generate([prompt], sampling_params)
    generation_time = time.time() - generation_time

    return outputs[0].outputs[0].text
