# \<Notice\>

본 시스템은 Raspberry Pi5 ‘4대’로 작동하는 시스템입니다.

아래 그림과 같이 준비해주시고, 각 보드에 각각 git clone 혹은 첨부파일 압축해제를 진행해주세요.

모든 보드는 같은 인터넷망을 사용해야 합니다.

<img width="1288" height="445" alt="image" src="https://github.com/user-attachments/assets/d0f25e36-5777-4ece-8b41-5f6480051b11" />


```bash
(git clone을 하는 경우)
git clone https://github.com/Park-Yul/Project2-YOLO26s-on-Raspberry-Pi-5.git

(unzip을 하는 경우)
unzip Team6_Project_On_Device_AI.zip
```

# 1. requirement.txt

우선 본인의 가상환경을 실행합니다.

```bash
(예시)
source ~/work/env/bin/activate
```

아래 명령어를 실행하여 requirements.txt에 첨부된 pip list를 다운받습니다.

```bash
pip install -r requirements.txt
```

# 2. door_master.py

상단 시스템도를 참고하여, ‘door_masker.py’가 써져있는 보드에서 다음 지침을 실행합니다.

우선 아래 명령어를 실행하여 vim을 엽니다.

```bash
vi door_master.py
```

vim에서 아래 명령어를 실행하여 다음 코드를 찾습니다.

```bash
/WORKERS =
```

<img width="467" height="83" alt="image" src="https://github.com/user-attachments/assets/e4a3b3cc-290e-4a45-96da-810ddf996df6" />


상단 시스템도를 참고하여, ‘door_worker.py’ 가 써져있는 두 보드의 IP 주소로 수정합니다.

<img width="468" height="81" alt="image" src="https://github.com/user-attachments/assets/41a2e574-0819-4d44-8300-5ebc4a1592f9" />


vim 저장 후(:wq) 터미널에서 아래 명령어를 통해 door_master.py를 실행 후 대기합니다.

```bash
python door_master.py
```

# 3. door_worker.py

상단 시스템도를 참고하여, ‘door_worker.py’가 써져있는 보드에서 다음 지침을 실행합니다.

우선 아래 명령어를 실행하여 vim을 엽니다.

```bash
vi door_worker.py
```

vim에서 아래 명령어를 실행하여 다음 코드를 찾습니다.

```bash
/MASKER =
```

<img width="481" height="20" alt="image" src="https://github.com/user-attachments/assets/c9529dbc-8f3f-4b73-8bcd-94ec84c63a8b" />


상단 시스템도를 참고하여, ‘door_masker.py’와 ‘app.py’가 써져있는 두 보드의 IP 주소로 수정합니다.

<img width="474" height="20" alt="image" src="https://github.com/user-attachments/assets/7f32af05-7670-4dc4-9507-d7785cddc546" />


vim 저장 후(:wq) 터미널에서 아래 명령어를 통해 door_worker.py를 실행 후 대기합니다.

```bash
python door_worker.py
```

# 4. app.py

상단 시스템도를 참고하여, ‘door_masker.py’와 ‘app.py’가 써져있는 보드에서 다음 지침을 실행합니다.

우선 아래 명령어를 실행하여 vim을 엽니다.

```bash
vi app.py
```

vim에서 아래 명령어를 실행하여 다음 코드를 찾습니다.

```bash
/PI_HOST =
```

<img width="422" height="23" alt="image" src="https://github.com/user-attachments/assets/1bec9fd6-272a-4119-a0ac-015495f05b4c" />


현재 보드의 IP 주소로 수정합니다.

<img width="416" height="19" alt="image" src="https://github.com/user-attachments/assets/12f1a1c6-7ab6-4e0d-ac36-ad96c3ac61f6" />


vim 저장 후(:wq) 터미널에서 아래 명령어를 통해 app.py를 실행 후 대기합니다.

```bash
python app.py
```

# 5. 웹브라우저에서 접속

http://[app.py 를 실행한 보드 IP]:8000 으로 접속합니다.

(PC와 보드는 같은 인터넷망을 사용하여야 합니다.)

<img width="620" height="701" alt="image" src="https://github.com/user-attachments/assets/f57ef5f2-2f28-4dd4-ae30-91e6e478e3f7" />


로그인 후 ‘실행’ 버튼을 누르면 door_master.py가 실행되며 모니터링이 시작됩니다.

<img width="777" height="586" alt="image" src="https://github.com/user-attachments/assets/98b48e14-a314-46a7-8f7a-5358bb9508f9" />

<img width="832" height="862" alt="image" src="https://github.com/user-attachments/assets/b0744b67-498f-4b9e-84d8-cc7c7b20730e" />


