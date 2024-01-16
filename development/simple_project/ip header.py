def calculate_checksum(header):
    # 将每两个字符组合成一个字节
    bytes_list = [int(header[i:i+2], 16) for i in range(0, len(header), 2)]

    # 计算累加和
    checksum = sum((bytes_list[i] << 8) + bytes_list[i+1] for i in range(0, len(bytes_list), 2))

    # 将溢出的部分加到低位
    while checksum >> 16:
        checksum = (checksum & 0xFFFF) + (checksum >> 16)

    # 取反码
    checksum = ~checksum & 0xFFFF

    return checksum

# 提供 IPv4 头部信息（示例）
ipv4_header = "4500003c4d8e400080010100030101000101"

# 计算校验和
result_checksum = calculate_checksum(ipv4_header)

# 输出结果
print("IPv4 头部校验和: {:04x}".format(result_checksum))
