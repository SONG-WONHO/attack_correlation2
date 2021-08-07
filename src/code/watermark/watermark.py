def get_watermark_dataset(config, X_train, y_train, X_test, y_test):

    if config.wm_type == "content":
        X_wm, y_wm = wm_content(config, X_train, y_train)

    return X_wm, y_wm, X_wm, y_wm


def wm_content(config, X, y):
    return X, y
