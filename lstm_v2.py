import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mplfinance as mpf
from tensorflow.keras import layers, models, optimizers
import tensorflow as tf


plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings(
    "ignore",
    message="Glyph.*missing from font\\(s\\) DejaVu Sans\\."
)


def preprocess_features(df):
    """计算技术指标"""
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma30'] = df['close'].rolling(window=30).mean()

    # 计算MACD指标
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['hist'] = df['macd'] - df['signal']

    # 计算KDJ指标
    low_min = df['low'].rolling(window=9).min()
    high_max = df['high'].rolling(window=9).max()
    df['rsv'] = (df['close'] - low_min) / (high_max - low_min) * 100
    df['k'] = df['rsv'].ewm(com=2).mean()
    df['d'] = df['k'].ewm(com=2).mean()
    df['j'] = 3 * df['k'] - 2 * df['d']

    # 计算波动率
    df['volatility'] = df['close'].rolling(window=20).std()

    # 计算相对强弱指标
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df.dropna(inplace=True)
    return df


def normalize_window(window):
    """使用窗口第一个值进行归一化，突出相对变化"""
    normalized = np.zeros_like(window)
    for i in range(window.shape[1]):
        if window[0, i] != 0:  # 避免除零错误
            normalized[:, i] = window[:, i] / window[0, i] - 1
    return normalized


def build_normalized_sequences(df, window_len, features):
    """构建归一化的序列数据"""
    arr = df[features].values
    X, metas = [], []

    for i in range(len(arr) - window_len + 1):
        window = arr[i:i + window_len]
        normalized = normalize_window(window)
        X.append(normalized)
        metas.append((df.index[i], df.index[i + window_len - 1]))

    return np.stack(X), metas


def hybrid_lstm_autoencoder(
        X,
        latent_dim=32,
        epochs=50,
        batch_size=64
):
    """混合LSTM自编码器，用于学习更有意义的特征表示"""
    T, F = X.shape[1], X.shape[2]

    inp = layers.Input(shape=(T, F))

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2))(inp)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True, dropout=0.2))(x)

    attention = layers.Dense(1, activation='tanh')(x)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(32 * 2)(attention)
    attention = layers.Permute([2, 1])(attention)

    x = layers.Multiply()([x, attention])
    x = layers.Lambda(lambda xin: tf.keras.backend.sum(xin, axis=1))(x)
    encoded = layers.Dense(latent_dim, activation='relu')(x)

    # 解码器
    x = layers.RepeatVector(T)(encoded)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True, dropout=0.2))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2))(x)
    decoded = layers.TimeDistributed(layers.Dense(F))(x)

    autoencoder = models.Model(inputs=inp, outputs=decoded)
    encoder = models.Model(inputs=inp, outputs=encoded)

    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.9
    )

    autoencoder.compile(optimizer=optimizers.Adam(learning_rate=lr_schedule), loss='mse')

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]

    val_size = max(1, int(len(X) * 0.1))
    autoencoder.fit(
        X[:-val_size], X[:-val_size],
        validation_data=(X[-val_size:], X[-val_size:]),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )

    return autoencoder, encoder


def calculate_correlation_score(target, candidate, features):
    """计算目标窗口和候选窗口之间的相关系数分数"""
    corrs = [np.corrcoef(target[f], candidate[f])[0, 1] for f in features]
    return np.nanmean(corrs)


def find_similar_kline_patterns_hybrid(
        df: pd.DataFrame,
        target_start: str,
        target_end: str,
        top_n: int = 3,
        min_days: int = 50,
        alpha: float = 0.5  # 混合权重
) -> pd.DataFrame:
    """
    使用混合方法返回不与目标区间重叠、且彼此不重叠的 Top-N 相似窗口。
    """

    # 目标区间
    target = df.loc[target_start:target_end]
    L = len(target)
    if L < min_days:
        raise ValueError(f"目标区间仅 {L} 日，不足 {min_days} 日。")

    t0 = pd.to_datetime(target_start)
    t1 = pd.to_datetime(target_end)

    # 基础特征
    base_feats = ['open', 'high', 'low', 'close', 'ma5', 'ma30']

    # 扩展特征
    all_feats = ['open', 'high', 'low', 'close', 'ma5', 'ma30', 'macd', 'hist', 'k', 'd', 'j', 'volatility', 'rsi']
    X_all, metas = build_normalized_sequences(df, L, all_feats)
    X_target = normalize_window(target[all_feats].values)[np.newaxis, ...]

    # 训练自编码器
    autoenc, encoder = hybrid_lstm_autoencoder(X_all, latent_dim=32, epochs=30)

    # 获取编码表示
    Z_all = encoder.predict(X_all, batch_size=128)
    Z_target = encoder.predict(X_target)

    # 计算相似度
    candidates = []
    for i in range(len(df) - L + 1):
        s, e = metas[i]
        # 跳过和目标区间有任何重叠的窗口
        if not (e < t0 or s > t1):
            continue

        # 计算相关系数得分
        candidate = df.iloc[i:i + L]
        corr_score = calculate_correlation_score(target, candidate, base_feats)

        # 计算自编码器特征相似度
        latent_score = np.dot(Z_all[i], Z_target[0]) / (np.linalg.norm(Z_all[i]) * np.linalg.norm(Z_target[0]))

        # 混合得分
        hybrid_score = alpha * corr_score + (1 - alpha) * latent_score

        candidates.append({'start': s, 'end': e, 'score': hybrid_score})

    cand_df = pd.DataFrame(candidates).sort_values('score', ascending=False)
    selected = []
    for _, row in cand_df.iterrows():
        s, e = row['start'], row['end']
        # 检查和已选所有片段都不重叠
        overlap = False
        for prev in selected:
            if not (e < prev['start'] or s > prev['end']):
                overlap = True
                break
        if overlap:
            continue
        selected.append({'start': s, 'end': e, 'score': row['score']})
        if len(selected) >= top_n:
            break

    return pd.DataFrame(selected)


def visualize_similar_kline(
        df: pd.DataFrame,
        target_start: str,
        target_end: str,
        top_n: int = 3,
        alpha: float = 0.5
):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    matches = find_similar_kline_patterns_hybrid(
        df=df,
        target_start=target_start,
        target_end=target_end,
        top_n=top_n,
        alpha=alpha
    )

    total = 1 + len(matches)
    fig, axes = plt.subplots(
        total, 1,
        figsize=(14, 3.5 * total),
        sharex=True,
        constrained_layout=True
    )

    for ax in axes:
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)

    # 目标区间
    tgt = df.loc[target_start:target_end]
    mpf.plot(
        tgt, type='candle', ax=axes[0],
        addplot=[
            mpf.make_addplot(tgt['ma5'], ax=axes[0], width=1),
            mpf.make_addplot(tgt['ma30'], ax=axes[0], width=1),
        ],
        show_nontrading=False
    )
    axes[0].set_title(f"目标区间 {target_start} → {target_end}", fontsize=9)
    axes[0].set_ylabel('Price', fontsize=8)

    # 匹配区间
    for i, row in enumerate(matches.itertuples(), start=1):
        seg = df.loc[row.start:row.end]
        mpf.plot(
            seg, type='candle', ax=axes[i],
            addplot=[
                mpf.make_addplot(seg['ma5'], ax=axes[i], width=1),
                mpf.make_addplot(seg['ma30'], ax=axes[i], width=1),
            ],
            show_nontrading=False
        )
        axes[i].set_title(
            f"匹配 #{i} [{row.start.date()}→{row.end.date()}], score={row.score:.3f}",
            fontsize=9
        )
        axes[i].set_ylabel('', fontsize=0)

    axes[-1].set_xlabel('Date', fontsize=8)
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    tecent_data = pd.read_csv('hk0700allhis.csv')
    tecent_data = preprocess_features(tecent_data)

    visualize_similar_kline(
        df=tecent_data,
        target_start='2025-02-01',
        target_end='2025-05-27',
        top_n=3,
        alpha=0.5
    )