from Crypto.Cipher import AES
import base64

# 密鑰和 IV 應該根據您的實際密鑰選擇
key = b'your16bytekey123'  # AES-128需要16位元組長的密鑰
iv = b'your16bytekey123'   # IV 應與密鑰長度相同

cipher = AES.new(key, AES.MODE_CBC, iv)  # 使用CBC模式
data = "asdadasdafsdggdfgd54fg1d51fgdf1g5"

# 填充
pad_length = 16 - (len(data) % 16)
data += chr(pad_length) * pad_length

encrypted = cipher.encrypt(data.encode())  # 加密
encoded = base64.b64encode(encrypted)  # Base64 編碼

print(encoded.decode())