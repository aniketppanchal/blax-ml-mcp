import gradio as gr

from .tools import (
    deploy_model,
    generate_request_id,
    train_model,
    upload_and_profile_csv,
)

with gr.Blocks() as app:
    gr.Markdown(
        """
        <div
            style="
                display: flex;
                flex-direction: column;
                align-items: center;
                text-align: center;
            "
        >
            <div style="display: flex; align-items: center; gap: 15px">
                <img
                    src="https://res.cloudinary.com/djeszvhyx/image/upload/v1772734058/logo_lrd2ts.png"
                    width="150px"
                />
                <div>
                    <h1 style="font-size: 40px; margin: 0">BlaxML MCP</h1>
                    <span>
                        Autonomous Machine Learning, Powered by
                        <a
                            href="https://blaxel.ai"
                            target="_blank"
                            style="color: #ff8b3d; text-decoration: none"
                        >
                            Blaxel
                        </a>
                    </span>
                </div>
            </div>
        </div>

        [Open README](https://github.com/aniketppanchal/blax-ml-mcp/blob/main/README.md)
        """
    )

    with gr.Tabs():
        with gr.TabItem("🆔 Generate Request ID"):
            gr.Interface(
                fn=generate_request_id,
                inputs=[],
                outputs=[gr.JSON()],
            )

        with gr.TabItem("📊 Upload and Profile CSV"):
            gr.Interface(
                fn=upload_and_profile_csv,
                inputs=[gr.Textbox(), gr.Textbox()],
                outputs=[gr.JSON()],
            )

        with gr.TabItem("⚙️ Train Model"):
            gr.Interface(
                fn=train_model,
                inputs=[gr.TextArea()],
                outputs=[gr.JSON()],
            )

        with gr.TabItem("🚀 Deploy Model"):
            gr.Interface(
                fn=deploy_model,
                inputs=[gr.Textbox()],
                outputs=[gr.JSON()],
            )

if __name__ == "__main__":
    app.launch(mcp_server=True)
