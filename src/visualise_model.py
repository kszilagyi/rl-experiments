import json

from src.cloud_run import run


def main():
    with open('render/params.json') as f:
        params = json.load(f)
    run(params, [], render_only=True)


if __name__ == '__main__':
    main()
