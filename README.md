
## Environment setup

```
Conda env create -f requirements.yml -p ./.venv
```

## Dataset Generation

### JS interpreter 
Install [nodejs](https://github.com/nodesource/distributions)
Install Canvas

```
cd i2p/dataset/js
npm install canvas
```

Run generator

```
python i2p/dataset/gen_data.py
```