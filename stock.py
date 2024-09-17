import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

# 숫자를 포맷팅하는 함수
def num_format(value):
    if abs(value) >= 1_000_000_000:
        return f'{value / 1_000_000_000:,.1f}B'
    elif abs(value) >= 1_000_000:
        return f'{value / 1_000_000:,.1f}M'
    elif abs(value) >= 1_000:
        return f'{value / 1_000:,.1f}K'
    else:
        return f'{value:,.1f}'

# tkinter를 사용하여 파일 선택 UI 만들기
root = tk.Tk()
root.withdraw()  # Tkinter 창 숨기기

# 파일 선택 대화 상자 열기
file_path = filedialog.askopenfilename(title="엑셀 파일을 선택하세요", filetypes=[("Excel files", "*.xlsx *.xls")])

# 엑셀 파일의 디렉토리 경로 추출
output_dir = os.path.dirname(file_path)

# 엑셀 파일 로드 (시트 이름을 명시적으로 지정하지 않고 첫 번째 시트를 사용)
df = pd.read_excel(file_path)

# 날짜 열을 datetime 형식으로 변환하여 더 나은 시각화를 위해 처리
df['기간종료일'] = pd.to_datetime(df['기간종료일'], format='%Y/%m/%d')

# 날짜 열을 인덱스로 설정하여 편리한 시각화 지원
df.set_index('기간종료일', inplace=True)

# 날짜를 오름차순으로 정렬하여 누적 계산이 정확하게 되도록 처리
df.sort_index(inplace=True)

# 1. 누적 수익금과 누적 수익률 계산
# 누적 수익금: 손익금액의 누적 합
df['누적 수익금'] = df['손익금액'].cumsum()

# 누적 수익률: 각 기간의 수익률을 복리 방식으로 누적하여 계산 (요구 사항에 따라 단순 합산도 가능)
df['누적 수익률'] = ((1 + df['수익률'] / 100).cumprod() - 1) * 100  # %로 표시

# 2. 드로우다운 및 롤링 변동성 계산
# 누적 최대값
df['누적 최대값'] = df['누적 수익금'].cummax()
# 드로우다운 계산
df['드로우다운'] = (df['누적 수익금'] - df['누적 최대값']) / df['누적 최대값']
# 롤링 변동성 계산 (20일)
df['롤링 변동성'] = df['수익률'].rolling(window=20).std()

# 그래프를 HTML로 저장하는 함수
def save_plot_to_html(fig, filename):
    full_path = os.path.join(output_dir, filename)
    fig.write_html(full_path)
    print(f"그래프가 '{full_path}'에 저장되었습니다.")

# 3. 누적 수익률과 누적 수익금 (Y축 두 개)
fig1 = go.Figure()
# 누적 수익률
fig1.add_trace(go.Scatter(x=df.index, y=df['누적 수익률'], mode='lines+markers+text', name='누적 수익률', yaxis='y1',
                          text=[num_format(val) for val in df['누적 수익률']], textposition='top center'))
# 누적 수익금
fig1.add_trace(go.Scatter(x=df.index, y=df['누적 수익금'], mode='lines+markers+text', name='누적 수익금', yaxis='y2',
                          text=[num_format(val) for val in df['누적 수익금']], textposition='top center'))

fig1.update_layout(
    title='누적 수익률과 누적 수익금',
    xaxis=dict(title='날짜'),
    yaxis=dict(title='누적 수익률 (%)', side='left', showgrid=False),
    yaxis2=dict(title='누적 수익금', side='right', overlaying='y', showgrid=False),
    font=dict(family="Arial, sans-serif")  # 한글 지원
)

save_plot_to_html(fig1, 'cumulative_return_and_profit_corrected.html')

# 4. 일일 수익금과 수익률 (Y축 두 개)
fig2 = go.Figure()
# 일일 수익금
fig2.add_trace(go.Scatter(x=df.index, y=df['손익금액'], mode='lines+markers+text', name='일일 수익금', yaxis='y1',
                          text=[num_format(val) for val in df['손익금액']], textposition='top center'))
# 수익률
fig2.add_trace(go.Scatter(x=df.index, y=df['수익률'], mode='lines+markers+text', name='수익률', yaxis='y2',
                          text=[num_format(val) for val in df['수익률']], textposition='top center'))

fig2.update_layout(
    title='일일 수익금과 수익률',
    xaxis=dict(title='날짜'),
    yaxis=dict(title='일일 수익금', side='left', showgrid=False),
    yaxis2=dict(title='수익률 (%)', side='right', overlaying='y', showgrid=False),
    font=dict(family="Arial, sans-serif")  # 한글 지원
)

save_plot_to_html(fig2, 'daily_profit_and_return_rate.html')

# 5. 드로우다운과 롤링 변동성 (Y축 두 개)
fig3 = go.Figure()
# 드로우다운
fig3.add_trace(go.Scatter(x=df.index, y=df['드로우다운'], mode='lines+markers+text', name='드로우다운', yaxis='y1',
                          text=[num_format(val) for val in df['드로우다운']], textposition='top center'))
# 롤링 변동성
fig3.add_trace(go.Scatter(x=df.index, y=df['롤링 변동성'], mode='lines+markers+text', name='롤링 변동성', yaxis='y2',
                          text=[num_format(val) for val in df['롤링 변동성']], textposition='top center'))

fig3.update_layout(
    title='드로우다운과 롤링 변동성',
    xaxis=dict(title='날짜'),
    yaxis=dict(title='드로우다운', side='left', showgrid=False),
    yaxis2=dict(title='롤링 변동성', side='right', overlaying='y', showgrid=False),
    font=dict(family="Arial, sans-serif")  # 한글 지원
)

save_plot_to_html(fig3, 'drawdown_and_rolling_volatility.html')

# 6. 손익금액의 정규 분포 (Plotly로 다이나믹 그래프 생성, 오른쪽 Y축에 정규 분포)
mu, std = norm.fit(df['손익금액'])
fig4 = px.histogram(
    df, 
    x='손익금액', 
    nbins=30, 
    title=f'손익금액의 정규 분포 (평균 = {mu:.1f}, 표준편차 = {std:.1f})', 
    histnorm='density',
    color_discrete_sequence=['blue'],  # 바 색상 지정
)

# 히스토그램 막대에 외곽선을 추가
fig4.update_traces(marker=dict(line=dict(width=1, color='black')))

# 각 막대의 높이(X 값)를 레이블로 표시
fig4.update_traces(
    texttemplate='%{x:,.0f}',  # 천 단위로 쉼표를 포함하는 포맷 지정
    textposition='outside'
)


# 정규 분포 곡선을 오른쪽 Y축에 추가
fig4.add_trace(go.Scatter(
    x=np.linspace(df['손익금액'].min(), df['손익금액'].max(), 100),
    y=norm.pdf(np.linspace(df['손익금액'].min(), df['손익금액'].max(), 100), mu, std),
    mode='lines', name='정규 분포', line=dict(color='black'),
    yaxis='y2'
))

# 레이아웃 업데이트: 두 번째 Y축 추가
fig4.update_layout(
    xaxis_title='손익금액',
    yaxis=dict(title='밀도', side='left'),
    yaxis2=dict(title='정규 분포', overlaying='y', side='right', showgrid=False),
    font=dict(family="Arial, sans-serif"),  # 한글 지원
    barmode='overlay'
)

save_plot_to_html(fig4, 'investment_profit_distribution_dynamic_with_dual_y_axis_and_outline.html')

# 7. 수익률의 정규 분포 (Plotly로 다이나믹 그래프 생성, 오른쪽 Y축에 정규 분포)
mu, std = norm.fit(df['수익률'])
fig5 = px.histogram(
    df, 
    x='수익률', 
    nbins=30, 
    title=f'수익률의 정규 분포 (평균 = {mu:.1f}, 표준편차 = {std:.1f})', 
    histnorm='density',
    color_discrete_sequence=['green'],  # 바 색상 지정
)

# 히스토그램 막대에 외곽선을 추가
fig5.update_traces(marker=dict(line=dict(width=1, color='black')))

# 각 막대의 x 값 레이블로 표시
fig5.update_traces(texttemplate='%{x:.2f}', textposition='outside')  # x값 레이블로 표시

# 정규 분포 곡선을 오른쪽 Y축에 추가
fig5.add_trace(go.Scatter(
    x=np.linspace(df['수익률'].min(), df['수익률'].max(), 100),
    y=norm.pdf(np.linspace(df['수익률'].min(), df['수익률'].max(), 100), mu, std),
    mode='lines', name='정규 분포', line=dict(color='black'),
    yaxis='y2'
))

# 레이아웃 업데이트: 두 번째 Y축 추가
fig5.update_layout(
    xaxis_title='수익률 (%)',
    yaxis=dict(title='밀도', side='left'),
    yaxis2=dict(title='정규 분포', overlaying='y', side='right', showgrid=False),
    font=dict(family="Arial, sans-serif"),  # 한글 지원
    barmode='overlay'
)

save_plot_to_html(fig5, 'return_rate_distribution_dynamic_with_dual_y_axis_and_outline.html')

# 저장된 파일 경로를 표시
print(f"수정된 그래프들이 다음 경로에 저장되었습니다: {output_dir}")
