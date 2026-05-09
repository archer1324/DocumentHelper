import streamlit as st

from backend.enterprise.ops import record_feedback
from backend.enterprise.react_core import ReActEnterpriseAssistant
from backend.enterprise.types import UserContext

st.set_page_config(page_title="Enterprise ReAct Doc Assistant", layout="wide")
st.title("Enterprise ReAct Documentation Assistant")

if "assistant" not in st.session_state:
    st.session_state.assistant = None
if "assistant_init_error" not in st.session_state:
    st.session_state.assistant_init_error = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_context" not in st.session_state:
    st.session_state.user_context = UserContext(
        user_id="u-demo",
        departments=["general"],
        roles=["employee"],
    )

with st.sidebar:
    st.subheader("User Context")
    user_id = st.text_input("User ID", value=st.session_state.user_context.user_id)
    departments = st.text_input("Departments (comma separated)", value=",".join(st.session_state.user_context.departments))
    roles = st.text_input("Roles (comma separated)", value=",".join(st.session_state.user_context.roles))
    if st.button("Apply Context", use_container_width=True):
        st.session_state.user_context = UserContext(
            user_id=user_id.strip() or "u-demo",
            departments=[x.strip() for x in departments.split(",") if x.strip()],
            roles=[x.strip() for x in roles.split(",") if x.strip()],
        )
        st.success("User context updated")

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("引用来源", expanded=False):
                for c in msg["citations"]:
                    st.markdown(
                        f"- [{c['source']}#{c['section']}] department={c['department']} version={c['version']} effective={c['effective_time']}"
                    )

        if msg["role"] == "assistant" and msg.get("feedback_key"):
            col1, col2, col3 = st.columns([1, 1, 3])
            with col1:
                if st.button("有用", key=f"useful_{msg['feedback_key']}"):
                    record_feedback(msg.get("query", ""), msg["content"], useful=True)
                    st.success("已记录反馈")
            with col2:
                if st.button("无用", key=f"useless_{msg['feedback_key']}"):
                    record_feedback(msg.get("query", ""), msg["content"], useful=False)
                    st.warning("已记录反馈")
            with col3:
                reason = st.text_input("报错说明", key=f"reason_{msg['feedback_key']}")
                if st.button("提交报错", key=f"report_{msg['feedback_key']}"):
                    record_feedback(msg.get("query", ""), msg["content"], useful=False, reason=reason)
                    st.info("报错反馈已提交")

prompt = st.chat_input("请输入你的企业文档问题")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.assistant is None:
        with st.spinner("正在初始化企业助手（首次可能较慢）..."):
            try:
                st.session_state.assistant = ReActEnterpriseAssistant()
                st.session_state.assistant_init_error = ""
            except Exception as e:
                st.session_state.assistant_init_error = str(e)

    if st.session_state.assistant is None:
        err = st.session_state.assistant_init_error or "未知错误"
        with st.chat_message("assistant"):
            st.error("助手初始化失败，请确认 Ollama 服务和模型可用。")
            st.code(err)
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner("ReAct 正在检索并推理..."):
            result = st.session_state.assistant.answer(prompt, st.session_state.user_context)

        st.markdown(result["answer"])
        st.caption(f"Intent: {result['intent']} | Confidence: {result['confidence']:.3f}")
        if result.get("citations"):
            with st.expander("引用来源", expanded=True):
                for c in result["citations"]:
                    st.markdown(
                        f"- [{c['source']}#{c['section']}] department={c['department']} version={c['version']} effective={c['effective_time']}"
                    )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "query": prompt,
                "citations": result.get("citations", []),
                "feedback_key": str(len(st.session_state.messages)),
            }
        )
