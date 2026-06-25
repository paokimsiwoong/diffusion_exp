"""
---
title: Utility functions for DDPM experiment
summary: >
  Utility functions for DDPM experiment
---

# Utility functions for [DDPM](index.html) experiemnt
"""
import torch.utils.data

from pathlib import Path

# logging 라이브러리 사용해보기
import logging

def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)

# with 구문으로 매번 확실하게 파일을 닫으므로 안전
  # 그 대신 로그를 할 때마다 파일을 열고 닫으므로 느림
def pal_maker(path:Path):
  log_fpath = path / "log.txt"
  def print_and_log(txt:str):
    print(txt)
    with log_fpath.open("a", encoding="utf-8") as f:
      f.write(f"{txt}\n")

  return print_and_log

# 파일을 열어둔 상태로 계속 로그를 쓰기 때문에 빠름
  # 그러나 파일.close를 종료전에 호출하지 않으면 메모리 누수 발생 가능
  # @@@ + 파이썬 write는 효율을 위해 write가 호출 되도 바로 디스크에 실제로 쓰지 않고
  # 메모리(버퍼)에 모아두므로 실험 도중 에러가 발생해 강제 종료 시 아직 버퍼에 남아있는 로그들이 유실된다.
    # with을 쓰면 파일을 닫으면서 버퍼에 남은 데이터를 flush해서 써주므로 안전
# def pal_maker2(path:Path):
#   log_fpath = path / "log.txt"
#   file = log_fpath.open("a", encoding="utf-8")
#   def print_and_log(txt:str):
#     print(txt)
#     file.write(f"{txt}\n")

#   return print_and_log

def create_logger(path:Path):
  log_fpath = path / "log.txt"

  # 로깅을 담당할 로거 인스턴스 생성
  logger = logging.getLogger("ExpLogger")

  # 로거 레벨 설정
    # DEBUG, INFO, WARNING, ERROR, CRITICAL 순으로 심각한 이벤트
    # 지정한 레벨의 이벤트 로그와 그보다 심각한 이벤트 로그를 기록하게 된다
    # ex: WARNING으로 설정하면 WARNING, ERROR, CRITICAL은 기록되고 DEBUG, INFO는 무시
  logger.setLevel(logging.INFO)

  # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  # 로거에 추가하는 핸들러들은 로거 인스턴스에 기록되는 로그들을 핸들러로 지정한 곳에 보내주는 역할을 한다.

  # logging.StreamHandler는 console 출력
  logger.addHandler(logging.StreamHandler())

  # logging.FileHandler는 로그를 파일에 쓰는 역할
  logger.addHandler(logging.FileHandler(log_fpath, mode='a', encoding='utf-8'))

  # @@@ 핸들러 별로 setLevel 메소드를 써서 기록되는 이벤트 로그의 레벨을 다르게 할 수도 있다.
  # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



  # 로거 인스턴스의 info 메소드는 INFO 레벨 로그를 생성한다.
  # create_logger로 반환받은 객체(로거.info 메소드)를 이용해 INFO 레벨 로그를 기록
  return logger.info