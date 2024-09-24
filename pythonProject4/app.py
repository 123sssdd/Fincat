from flask import Flask, render_template, request, session,jsonify
import requests
from flask import session
from flask_caching import Cache
import re # 使用正则表达式

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 用于 session 管理

# 配置缓存为简单缓存，存储在内存中
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# 与后端交互
def get_answer():
    # 获取对话历史
    messages = session.get('messages', [])
    # 获取用户问题
    user_question = messages[-1]['content']
# 对问题进行 URL 编码
    encoded_question = requests.utils.quote(user_question, safe='')

    # 构建请求的 URL
    url = 'http://localhost:8001/question/' + encoded_question

    # 设置超时时间为30秒
    timeout = 30

    try:
        # 发送 GET 请求
        response = requests.get(url, timeout=timeout)

        # 检查响应状态码
        if response.status_code == 200:
            try:
                # 解析 JSON 响应
                json_response = response.json()

                company_name = json_response['classResult'][0]['公司名称']
                years = json_response['classResult'][0]['年份']
                years_str = ', '.join(years)
                # 提取 answer 内容
                answer_text = json_response['answer'][0]['answer']+"\n数据来源于"+company_name+years_str+"财报"

                # 返回答案
                return answer_text
            except (ValueError, KeyError, IndexError):
                # 无法解析服务器的响应
                return '无法解析服务器的响应。'
        else:
            # 服务器错误，无法获取答案
            return '服务器错误，无法获取答案。'
    except requests.Timeout:
        # 请求超时，请重试
        return '请求超时，请重试。'
    except requests.RequestException as e:
        # 发生其他请求异常
        return f'发生错误：{e}'



# 处理用户请求
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # 从表单中获取数据
        user_question = request.form.get('question')
        uploaded_file = request.files.get('file')

        # 处理文件上传（如有需要）
        if uploaded_file:
            # 处理上传的文件，例如保存或解析
            pass  # 添加你的文件处理逻辑

        # 更新会话消息
        if 'messages' not in session:
            session['messages'] = [
                {"role": "system", "content": "你现在是一只叫绿豆猫猫包的聪明小猫，拥有丰富的金融知识。"}
            ]

        # 将用户问题添加到会话
        session['messages'].append({"role": "user", "content": user_question})

        # 获取模型的回答
        answer = get_answer()

        # 将助手的回答添加到会话
        session['messages'].append({"role": "assistant", "content": answer})

        # 返回 JSON 响应
        return jsonify({'answer': answer})
    else:
        # GET 请求，渲染模板并传递消息历史
        messages = session.get('messages', [])
        return render_template('index.html', messages=messages)


# 重置对话历史
@app.route('/reset', methods=['POST'])
def reset():
    session.pop('messages', None)
    return jsonify({'status': '会话历史已清空'})


if __name__ == '__main__':
    app.run(debug=True)
