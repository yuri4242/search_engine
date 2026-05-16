from langchain_community.llms import LlamaCpp

def build_llm(config):
    cfg = config["llm"]
    return LlamaCpp(
            model_path=cfg["model_path"],
            n_ctx=cfg["n_ctx"],
            n_threads=cfg["n_threads"],
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
            stop=cfg["stop"],
            verbose=False,
            )
