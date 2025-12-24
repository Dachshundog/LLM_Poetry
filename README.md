# LLM_Peom

---------------------------------------------------------------------------------------------------
## abstract image Using LLM-based Agent
This project explores the potential of Large Language Models (LLMs) to model 
---------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------

## Setup Instructions

1. **Create and activate a Conda environment**

   ```bash
   conda create -n LLMPoetry python=3.10 -y
   conda activate LLMPoetry
   ```
 
2. **Install dependencies**
   下面这安装依赖的操作二选一即可

   ```bash
   pip install networkx pandas matplotlib openai openpyxl python-dotenv
   ```

   or install all from the lock file:

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API access with `.env`**

   直接打开文件里的env文件，填入你在火山引擎上开通的 OPENAI_API_KEY和ENDPOINT  （地址：https://console.volcengine.com/ark/region:ark+cn-beijing/endpoint?config=%7B%7D）
      
   3. Load environment variables once at startup (already handled in many setups):
      #主文件是LLM_Poem.py，运行前先在E:\Qianlong\LLM_poem_viewer\LLM_Client.py里,下面这句代码处修改env.文件地址

      ```python
         from dotenv import load_dotenv
         load_dotenv("E:\Qianlong\LLM_poem_viewer\.env") 
      ```

   > **Tip:** Never commit your `.env` file or any API keys to version control.

4. **Run an example**
   #运行前记得修改input json的path
   使用 VS Code 终端 或者打开anaconda在上面新建环境LLMPoetry里执行 ：
   ```bash
   python LLM_Poem.py --temperature 0.2 --max_poems 20   #默认 0.2, 检索类任务, 调试过程中只导出前20条
   ```
