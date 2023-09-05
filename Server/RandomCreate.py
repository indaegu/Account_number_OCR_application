# 최종수정일: 23.09.05 (화)
# 간단소개 : 계좌랜덤 생성 코드


import random
import string

# 은행 정보와 계좌 길이
bank_info = {
    '농협': 13,
    '신한': 12,
    '국민': 14,
    '기업': 12
}

# 은행 ID 매핑
bank_id_mapping = {
    '농협': 1,
    '기업': 2,
    '국민': 3,
    '신한': 4
}


# 한국인 이름의 첫 글자, 중간 글자, 마지막 글자 후보
first_names = ['김', '이', '박', '최', '정', '강', '조', '윤', '장', '임']
middle_names = ['민', '지', '유', '현', '승', '영', '상', '수', '기', '희']
last_names = ['수', '은', '우', '윤', '원', '영', '아', '훈', '오', '미']

# 고객 정보와 은행 계좌 정보를 저장할 리스트
customer_info_list = []
bank_account_list = []

# 생성된 계좌 번호를 저장할 집합 (중복 확인용)
generated_account_numbers = set()

# 100개의 랜덤한 튜플 생성
for i in range(1, 101):
    # 랜덤한 이름 생성
    name = random.choice(first_names) + random.choice(middle_names) + random.choice(last_names)

    # 랜덤한 은행 선택
    bank_name, account_length = random.choice(list(bank_info.items()))

    # 랜덤한 계좌 번호 생성 (중복되지 않도록)
    while True:
        account_number = ''.join(random.choices(string.digits,k=account_length))
        if account_number not in generated_account_numbers:
            generated_account_numbers.add(account_number)
            break

    # 고객 정보와 은행 계좌 정보 저장
    customer_info = (i, name, bank_name)
    bank_account = (account_number, i, bank_name)

    customer_info_list.append(customer_info)
    bank_account_list.append(bank_account)

# SQL 쿼리를 생성하기 위한 코드

customer_info_sql_queries = []
bank_account_sql_queries = []

# 고객 정보 쿼리 생성
for customer_id, name, bank_name in customer_info_list:
    customer_info_sql = f"INSERT INTO customer_info VALUES ({customer_id}, '{name}', '{bank_name}');"
    customer_info_sql_queries.append(customer_info_sql)

bank_account_sql_queries_updated = []
# 은행 계좌 정보 쿼리 생성
for account_number, customer_id, bank_name in bank_account_list:
    bank_id = bank_id_mapping[bank_name]  # 업데이트된 은행ID
    bank_account_sql = f"INSERT INTO bank_account VALUES ('{account_number}', {customer_id}, {bank_id}, 0);"
    bank_account_sql_queries_updated.append(bank_account_sql)

# 출력해 봅니다.
print("고객정보 출력")
for query in customer_info_sql_queries[:101]:
    print(query)
print("계좌정보 출력")
for query in bank_account_sql_queries_updated[:101]:
    print(query)



