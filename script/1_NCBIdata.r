

# 原始数据合并整理
library(tidyverse)


# fasta_preprocess = function(filename) {
  # Read the fasta file
# }

filename = "./data/sirt6_sequence_1307.fasta"
fasta = readLines(filename)

location = which(str_sub(fasta, 1, 1) == ">") # 定位新序列起始符号>

name = vector("character", length(location)) # 取出名称存入数组
sequence = vector("character", length(location)) # 初始化序列数组

for (i in 1:length(location)) {
  print(i)
  # Extract sequence name
  name_line = location[i]
  temp_name = fasta[name_line]
  name[i] = temp_name
  # Extract sequence
  seq_start = location[i] + 1
  if (i != length(location)) {
    seq_end = location[i + 1] - 1
  } else {
    seq_end = length(fasta)
  }
  seq_range = seq_start:seq_end
  temp_seq = paste(fasta[seq_range], collapse = "")
  sequence[i] = temp_seq
  # print(nchar(temp_seq))
}

sirt6_tb = tibble(name = name, sequence = sequence)

# Regular expression
# 结果中出现一些三个变量均为NA的数据，经检查应当确实没有种系等信息，无法利用，因此不再特别处理

formated_tb = sirt6_tb |>
  extract(
    col = "name",
    into = c("id", "name", "species"),
    regex = ">(\\S+)\\s(.*)\\s\\[(.*)\\]"
  ) |>
  mutate(species_short = str_extract(species, "(\\S+\\s\\S+)"))

write_csv(formated_tb, "./processed_data/sirt6_sequence_1307a.csv")


#############到此为止，因为整合寿命数据的原因，后面转python pandas处理

