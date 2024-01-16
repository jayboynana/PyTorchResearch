import urllib
from urllib.parse import urlencode
from urllib.request import urlopen

# 前半部分的链接(注意是http，不是https)
url_pre = 'http://www.baidu.com/s'

# GET参数
params = {}
params['wd'] = u'测试'.encode('utf-8')
url_params = urlencode(params)

# GET请求完整链接
url = '%s?%s' % (url_pre, url_params)

# 打开链接，获取响应
response = urlopen(url)
# response = urllib2.urlopen(url)

# 获取响应的html
html = str(response.read())

# 将html保存到文件
with open('test.txt', 'w') as f:
    f.write(html)