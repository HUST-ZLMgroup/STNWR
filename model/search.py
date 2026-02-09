import numpy as np
from math import sqrt
from tqdm.auto import tqdm
from copy import deepcopy
import time


def golden_section(a, c, A, C, tau_decimal, delta, function, tol, max_iter, int_score=False, verbose=False):
    """
    :param a: bw最小值
    :param c: bw最大值
    :param A: tau最小值
    :param C: tau最大值
    :param tau_decimal: tau有效数字位数
    """
    b = a + delta * np.abs(c - a)
    d = c - delta * np.abs(c - a)
    B = A + delta * np.abs(C - A)
    D = C - delta * np.abs(C - A)

    opt_score = np.inf
    diff = 1.0e9
    iters = 0
    search_dict = {}
    opt_bw, opt_tau = None, None
    while np.abs(diff) > tol and iters < max_iter and a != np.inf:
        iters += 1
        if int_score:
            b = np.round(b)
            d = np.round(d)
        else:
            b = np.round(b, 2)
            d = np.round(d, 2)
        B = np.round(B, tau_decimal)
        D = np.round(D, tau_decimal)
        if f'{b}_{B}' in search_dict:
            score_bB = search_dict[f'{b}_{B}']
        else:
            try:
                score_bB = function(b, B)
                search_dict[f'{b}_{B}'] = score_bB
                if verbose:
                    print(f"Bandwidth: {np.round(b, 2):10}, tau: {B:10.5f}, score: {score_bB:.3f}")
            except Exception as e:
                print(e)
                break

        if f'{b}_{D}' in search_dict:
            score_bD = search_dict[f'{b}_{D}']
        else:
            try:
                score_bD = function(b, D)
                search_dict[f'{b}_{D}'] = score_bD
                if verbose:
                    print(f"Bandwidth: {np.round(b, 2):10}, tau: {D:10.5f}, score: {score_bD:.3f}")
            except Exception as e:
                print(e)
                break

        if f'{d}_{B}' in search_dict:
            score_dB = search_dict[f'{d}_{B}']
        else:
            try:
                score_dB = function(d, B)
                search_dict[f'{d}_{B}'] = score_dB
                if verbose:
                    print(f"Bandwidth: {np.round(d, 2):10}, tau: {B:10.5f}, score: {score_dB:.3f}")
            except Exception as e:
                print(e)
                break

        if f'{d}_{D}' in search_dict:
            score_dD = search_dict[f'{d}_{D}']
        else:
            try:
                score_dD = function(d, D)
                search_dict[f'{d}_{D}'] = score_dD
                if verbose:
                    print(f"Bandwidth: {np.round(d, 2):10}, tau: {D:10.5f}, score: {score_dD:.3f}")
            except Exception as e:
                print(e)
                break

        tmp_min = min(score_bB, score_bD, score_dB, score_dD)
        opt_score = tmp_min
        if score_bB == tmp_min:
            opt_bw = b
            c = d
            d = b
            b = a + delta * np.abs(c - a)
            opt_tau = B
            C = D
            D = B
            B = A + delta * np.abs(C - A)
        elif score_bD == tmp_min:
            opt_bw = b
            c = d
            d = b
            b = a + delta * np.abs(c - a)
            opt_tau = D
            A = B
            B = D
            D = C - delta * np.abs(C - A)
        elif score_dB == tmp_min:
            opt_bw = d
            a = b
            b = d
            d = c - delta * np.abs(c - a)
            opt_tau = B
            C = D
            D = B
            B = A + delta * np.abs(C - A)
        else:
            opt_bw = d
            a = b
            b = d
            d = c - delta * np.abs(c - a)
            opt_tau = D
            A = B
            B = D
            D = C - delta * np.abs(C - A)

        opt_bw = np.round(opt_bw, 2)
        opt_tau = np.round(opt_tau, tau_decimal)
        diff = sqrt((score_bB - score_dD) ** 2 + (score_bD - score_dB) ** 2)

    if opt_bw is None or opt_tau is None:
        raise Exception('opt_val or opt_tau is None!')
    return (opt_bw, opt_tau), opt_score, search_dict


def equal_interval(a, c, bw_interval,
                   A, C, tau_interval,
                   function, int_score=False, verbose=False):
    if int_score:
        a = np.round(a)
        c = np.round(c)
        bw_interval = np.round(bw_interval)
    if bw_interval <= 0 or tau_interval <= 0:
        raise Exception('bw_interval <= 0 or tau_interval <= 0')

    search_dict = {}
    opt_score = np.inf
    opt_bw, opt_tau = None, None
    b = a
    while b <= c:
        B = A
        while B <= C:
            try:
                score_bB = function(b, B)
            except Exception as e:
                print(e, b, B)
                b = b + bw_interval
                B = B + tau_interval
                continue
            if verbose:
                print(f"Bandwidth:{b:10}, tau:{B:10.5f}, score:{score_bB:.3f}")

            if score_bB < opt_score:
                opt_bw = b
                opt_tau = B
                opt_score = score_bB
            search_dict[f'{b}_{B}'] = score_bB
            B = B + tau_interval
        b = b + bw_interval

    if opt_bw is None or opt_tau is None:
        raise Exception('opt_val or opt_tau is None!')
    return (opt_bw, opt_tau), opt_score, search_dict


def multi_bw(multi_init, y, X, n, k, tol, max_iter, rss_score, gtwr_func, sel_func,
             multi_bw_min, multi_bw_max, max_same_times, verbose=False):
    bw, tau, err, param = multi_init
    y = y.reshape((-1, 1))
    err = err.flatten()
    Y_pred = np.multiply(param, X)
    rss = np.sum(err ** 2)
    scores = []
    BWs_TAUs = []
    stable_counter = 0
    bws_taus = np.empty((k, 2))

    new_Y_pred = np.zeros_like(X)
    params = np.zeros_like(X)
    for iters in tqdm(range(1, max_iter + 1), desc='Backfitting'):
        for j in range(k):
            temp_y = Y_pred[:, j]
            temp_y = temp_y + err
            temp_X = X[:, j]
            if stable_counter >= max_same_times:
                bw, tau = bws_taus[j]
            else:
                bw, tau = sel_func(temp_y, temp_X, multi_bw_min[j], multi_bw_max[j])
            optim_model = gtwr_func(temp_y, temp_X, bw, tau)
            err = optim_model.resid_response
            param = optim_model.params
            new_Y_pred[:, j] = optim_model.predy
            params[:, j] = param
            bws_taus[j] = bw, tau
        if (iters > 1) and np.all(BWs_TAUs[-1] == bws_taus):
            stable_counter += 1
        else:
            stable_counter = 0

        num = np.sum((new_Y_pred - Y_pred) ** 2) / n
        den = np.sum(np.sum(new_Y_pred, axis=1) ** 2)
        score = (num / den) ** 0.5
        Y_pred = new_Y_pred

        if rss_score:
            predy = np.sum(np.multiply(params, X), axis=1)
            new_rss = np.sum((y - predy) ** 2)
            score = np.abs((new_rss - rss) / new_rss)
            rss = new_rss
        scores.append(deepcopy(score))
        delta = score
        BWs_TAUs.append(deepcopy(bws_taus))

        if verbose:
            print("Current iteration:", iters, ",SOC:", np.round(score, 7))
            print("Bandwidths:", ', '.join([str(bw) for bw in bws_taus[:, 0].reshape(-1)]))
            print("taus:", ', '.join([str(bw) for bw in bws_taus[:, 1].reshape(-1)]))

        if delta < tol:
            break

    opt_bws_taus = BWs_TAUs[-1]
    return opt_bws_taus, np.array(BWs_TAUs), np.array(scores), params, err
