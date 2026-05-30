from blaxel.core import SandboxInstance

from .config import settings


class SandboxManager:
    MODEL_TRAINER_SANDBOX_NAME = settings.model_trainer_sandbox_name
    MODEL_TRAINER_SANDBOX_IMAGE = settings.model_trainer_sandbox_image
    MODEL_TRAINER_SANDBOX_MEMORY = settings.model_trainer_sandbox_memory

    MODEL_SERVER_SANDBOX_IMAGE = settings.model_server_sandbox_image
    MODEL_SERVER_SANDBOX_MEMORY = settings.model_server_sandbox_memory
    MODEL_SERVER_SANDBOX_PORT = 7860
    MODEL_SERVER_SANDBOX_TTL = settings.model_server_sandbox_ttl

    @classmethod
    async def get_trainer_sandbox(cls) -> SandboxInstance:
        return await SandboxInstance.create_if_not_exists(
            {
                "name": cls.MODEL_TRAINER_SANDBOX_NAME,
                "image": cls.MODEL_TRAINER_SANDBOX_IMAGE,
                "memory": cls.MODEL_TRAINER_SANDBOX_MEMORY,
                "region": settings.bl_region,
            }
        )

    @classmethod
    async def create_server_sandbox_with_preview(
        cls,
        request_id: str,
    ) -> tuple[SandboxInstance, str]:
        sandbox = await SandboxInstance.create_if_not_exists(
            {
                "name": request_id,
                "image": cls.MODEL_SERVER_SANDBOX_IMAGE,
                "memory": cls.MODEL_SERVER_SANDBOX_MEMORY,
                "region": settings.bl_region,
                "ports": [{"target": cls.MODEL_SERVER_SANDBOX_PORT}],
                "ttl": cls.MODEL_SERVER_SANDBOX_TTL,
            }
        )
        preview = await sandbox.previews.create_if_not_exists(
            {
                "metadata": {"name": "public-preview"},
                "spec": {"port": cls.MODEL_SERVER_SANDBOX_PORT, "public": True},
            }
        )
        return sandbox, preview.spec.url
