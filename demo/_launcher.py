# 데모 실행용 로더입니다. 완성본 코드는 컴파일되어 있습니다 — 직접 만들어 보세요!
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# import가 아니라 runpy로 실행해야 Streamlit이 재실행할 때마다 화면이 다시 그려진다.
runpy.run_path(str(HERE / "demo_app.pyc"), run_name="__main__")
