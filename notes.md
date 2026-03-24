`string.encode("utf-8")`: `str` 转换成 `bytes`类型

`bytes`类型可以迭代 / 解包，自动转换成 int：`list(bytes)` 

`bytes.decode("utf-8")`：`bytes`类型转换成`str`

## bytes() 函数

创建一个**不可变的字节序列**

| **传入参数类型**            | **示例**                 | **结果**                                                     |
| --------------------------- | ------------------------ | ------------------------------------------------------------ |
| **字符串 (str)**            | `bytes("你好", "utf-8")` | 将字符按指定编码转为字节。**必须指定编码**。                 |
| **整数 (int)**              | `bytes(5)`               | 创建一个指定长度、内容全为 `0`（空位）的字节串。             |
| **可迭代对象 (list/tuple)** | `bytes([65, 66, 67])`    | 将 0-255 的整数列表转为对应的 ASCII 字节（此处为 `b'ABC'`）。 |
| **无参数**                  | `bytes()`                | 创建一个长度为 0 的空字节对象。                              |

注意：如果传入的是一个`int n`，含义是创建一个长度为`n`的全0(`\x00`)空串

## Problem2: Unicode Encodings

#### (a) 为什么 tokenizer 最偏好使用`UTF-8`而不是`UTF-16`或`UTF-32`

1. 空间效率和稀疏性：`UTF-8` 只用 `1bytes` 可以表示标准的`ASCII字符`，后两者需要 `2/4 bytes`；因此：
   1. `UTF-8`**内存占用更低**（英语文本和编程语言）；
   2. **词语存储效率(Vocabulary Efficiency)更高**："a" 在 UTF-32中，会占用三个空字符来表示，效率很低。
2. 对空字符(Null Byte)的鲁棒性更好：由于UTF-16/32有大量的 00 字节，而大量C-based libraries以及legacy data processing tools将 00 视作字符串结尾标志。这会导致错误的数据截断和意外崩溃；
3. 连续性和一致性：UTF-8与字节流兼容，如果从文件中间开始读文件，UTF-8解码器通常能够找到下一个有效字符的开头（每个字节的“头部”位指示他是起始字节还是延续字节）。而`UTF-16/32`对字节序（大端/小端）很敏感，如果发送数据的系统和训练模型读字节的顺序不同，整个数据集就成为乱码。
4. UTF-8 标准是大部分网页（98%）和大部分数据集的编码格式；

#### (b) 如下这个将`UTF-8`编码转换成Unicode string的函数错在哪里？

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
	return "".join([bytes([b]).decode("utf-8") for b in bytestring])

>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```

我们来看这个函数在做什么，首先来看列表生成式内部：

1. 遍历 `bytestring`，得到每一个byte的十进制表示 `b: int`；
2. 将其装入迭代器`[b]`，然后用`bytes()`函数将其转换成`bytes`；
3. 最后，将得到的`bytes` decode，得到`str`

然后将这个`str`的拼接成一个字符串。

这个函数错在哪里？在ASCII编码中，每个字符确实只占`1B`，但`UTF-8`是变长编码（`1~4 B`），第一个字节的前几个字符或指明其长度。如果这里传入非ASCII编码所涵盖的字符，这里就会解码出错。比如：

```python
>>> decode_utf8_bytes_to_str_wrong("你好👋".encode("utf-8"))
Traceback (most recent call last):
  File "<python-input-10>", line 1, in <module>
    decode_utf8_bytes_to_str_wrong("你好👋".encode("utf-8"))
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<python-input-0>", line 2, in decode_utf8_bytes_to_str_wrong
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
                    ~~~~~~~~~~~~~~~~~^^^^^^^^^
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe4 in position 0: unexpected end of data
```

报了一个`0xe4`不能出现在第一个byte的错。

#### (C) 给出一个 2 btye sequence，不能被decode成任何Unicode字符

Unicode 规则的核心是前缀码机制，通过字节最左侧的比特位，解码器能够立即判断这是一个几字节的字符，或者这是一个后跟的跟随字节：

```python
1字节(ASCII)：0xxxxxxx
2字节				：110xxxxx 10xxxxxx
3字节				：1110xxxx 10xxxxxx 10xxxxxx
4字节				：11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
```

1字节 0 开头，保证能够正确继承所有`ASCII字符`。

只需要随意写出一个不满足上述规则的序列即可，如`b'\xc2\x20'`

```python
>>> b"\xc2\x20".decode("utf-8")
Traceback (most recent call last):
  File "<python-input-26>", line 1, in <module>
    b"\xc2\x20".decode("utf-8")
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc2 in position 0: invalid continuation byte
```

