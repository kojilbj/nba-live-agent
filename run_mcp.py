import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from nba_live_agent.mcp_server import mcp  # noqa: E402

if __name__ == "__main__":
    mcp.run()
