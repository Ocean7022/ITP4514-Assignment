# How to use
Please install `Poetry` to set up a virtual environment

If not, please install

### Install on `Windows(Powershell)`
``` 
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

### Set path on `Windows(Powershell)`
```
$Env:Path += ";C:\Users\<Your User Name>\AppData\Roaming\Python\Scripts"; setx PATH "$Env:Path"
```
Don’t forget to replace `<Your User Name>` with your username!

#

### Install on `Linux, macOS, Windows(WSL)`
```
curl -sSL https://install.python-poetry.org | python3 -
```

### Set path on `Linux, macOS, Windows(WSL)`
```
export PATH=$PATH:$HOME/.local/bin
```

You can see more about `Poetry` on [Poetry Website](https://python-poetry.org/docs/)
# Install Google Chome
[How to Install Google Chrome Using Terminal on Linux](https://www.wikihow.com/Install-Google-Chrome-Using-Terminal-on-Linux)
# Set up virtual environment
### Enter project root directory
```
cd <Your path>/webDataCawler
```
### Create virtual environment
```
poetry shell
```
Now you can see that the virtual environment has been created

Your command line should look like this

`(webdatacawler-py3.11) PS C:\Users\...`

### Install requirements packages
```
poetry install
```

# Replace `middlewares.py`

Replace the `middlewares.py` code under the  scrapy_selenium package

`<Your venv path>/Lib/site-packages/scrapy_selenium/middlewares.py`

### Check `<Your venv path>` 
```
poetry env info --path
```

[middlewares.py](https://github.com/clemfromspace/scrapy-selenium/blob/5c3fe7b43ab336349ef5fdafe39fc87f6a8a8c34/scrapy_selenium/middlewares.py)
It also provided in `installation` directory

[Solution source](https://github.com/clemfromspace/scrapy-selenium/issues/128)

# Run spiders
### Enter webDataCawler directory
```
cd /source/webDataCawler/
```

### Run spider
```
scrapy crawl <Spider Name> -a path=<Your target path>
```

### Example:
```
scrapy crawl billion_tw -a path=C:\Users\ocean\results
```
If the directory does not exist, the directory will create itself

# Spiders
`aspeedtech_tw` - https://www.aspeedtech.com/tw/news

`billion_tw` - https://www.billion.com/zh-tw

`mxic_tw` - https://www.mxic.com.tw

`newbest_tw` - https://www.newbest.com.tw/en/

`realtek_tw` - https://www.realtek.com/zh-tw/press-room/news-releases