from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate

# 启动 LLM 服务
llm = LlamaCpp(
    model_path=r"C:\RAG\RAG_example\Model_GGUF\Llama3-8B-Chinese-Chat-q8-v2.gguf",
    n_gpu_layers=100,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)

# 创建聊天模板
chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个专业且有帮助的AI金融顾问。你的名字是劉道恩。"),
        ("human", "你好，你今天怎么样？"),
        ("ai", "這個問題你問過了，笨蛋"),
        ("human", "{user_input}"),
    ]
)

# 创建 PromptChain 实例
compliance_prompt = PromptTemplate(
    input_variables=["scenario"],
    template="""根据所提供的场景和金融法规数据库，判断描述的情况是否符合相关法规。如果不符合，请说明基于具体法规的原因，并提供合规建议。场景：{scenario}"""
)
prompt_chain = LLMChain(prompt=compliance_prompt, llm=llm)

# 初始问候
print("您好！我是您的金融顾问劉道恩，很高兴为您提供帮助。请问有什么我可以为您做的？")

# 用户界面 - 动态输入查询
while True:
    query = input("请输入您的问题，或者输入 'exit' 退出: ")
    if query.lower() == 'exit':
        print("感谢您的咨询。如果您有其他问题，随时联系我。祝您生活愉快，再见！")
        break
    
    # 使用聊天模板生成响应
    formatted_message = chat_template.format(user_input=query)
    response = llm.invoke(formatted_message)
    
    # 生成专业且安定的回复
    response_text = f"根据您的问题，我找到了一些信息：{response}\n如果您有其他问题，请随时告诉我。我在这里为您提供帮助。"
    print("回应：", response_text)
