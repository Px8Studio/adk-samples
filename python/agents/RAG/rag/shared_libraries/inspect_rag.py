import vertexai.preview.rag as rag
import inspect

print("Available functions in vertexai.preview.rag:")
for name, obj in inspect.getmembers(rag):
    if inspect.isfunction(obj) or inspect.isclass(obj):
        if not name.startswith("_"):
            print(f"- {name}")
