import pandas as pd
import numpy as np
from scipy.stats import norm


def simulate_periodic_review_L0(
    mu_demand,
    std_demand,
    review_period,
    service_level,
    initial_inventory,
    demand_series,
    holding_cost=0.0,
    shortage_cost=0.0,
    order_cost=0.0,
    return_details=True
):
    """
    Mô phỏng hệ thống tồn kho periodic review với:
    - Lead time L = 0
    - Safety stock được tính theo công thức lý thuyết
    - Order-up-to level S được tính theo công thức lý thuyết
    - demand_series dùng để kiểm thử mô phỏng

    Parameters
    ----------
    mu_demand : float
        Nhu cầu trung bình mỗi kỳ.
    std_demand : float
        Độ lệch chuẩn nhu cầu mỗi kỳ.
    review_period : int
        Chu kỳ kiểm tra tồn kho R.
    service_level : float
        Mức độ phục vụ mong muốn, ví dụ 0.95.
    initial_inventory : float
        Tồn kho đầu kỳ.
    demand_series : list, np.ndarray, pd.Series
        Chuỗi nhu cầu thực tế dùng để chạy mô phỏng.
    holding_cost : float, default=0.0
        Chi phí lưu kho trên mỗi đơn vị tồn cuối kỳ.
    shortage_cost : float, default=0.0
        Chi phí thiếu hàng trên mỗi đơn vị thiếu.
    order_cost : float, default=0.0
        Chi phí cố định cho mỗi lần đặt hàng.
    return_details : bool, default=True
        Nếu True trả về bảng chi tiết từng kỳ.

    Returns
    -------
    results_df : pd.DataFrame or None
        Bảng mô phỏng chi tiết theo từng kỳ.
    summary : dict
        Thống kê tổng hợp mô phỏng.
    """

    if review_period <= 0:
        raise ValueError("review_period phải > 0")
    if std_demand < 0:
        raise ValueError("std_demand không được âm")
    if not (0 < service_level < 1):
        raise ValueError("service_level phải nằm trong khoảng (0, 1)")
    if initial_inventory < 0:
        raise ValueError("initial_inventory không được âm")

    demand_series = pd.Series(demand_series, dtype=float).reset_index(drop=True)
    if len(demand_series) == 0:
        raise ValueError("demand_series không được rỗng")

    # =========================
    # 1. TÍNH POLICY THEO LÝ THUYẾT
    # =========================
    z = norm.ppf(service_level)

    # Vì L = 0 nên protection period = R
    sigma_R = std_demand * np.sqrt(review_period)
    safety_stock = z * sigma_R
    expected_demand_over_review = mu_demand * review_period
    order_up_to_level = expected_demand_over_review + safety_stock

    # =========================
    # 2. MÔ PHỎNG
    # =========================
    inventory = float(initial_inventory)

    total_holding_cost = 0.0
    total_shortage_cost = 0.0
    total_order_cost = 0.0
    total_order_qty = 0.0
    total_shortage_units = 0.0
    total_sales = 0.0
    num_orders = 0

    records = []

    for t, demand in enumerate(demand_series, start=1):
        beginning_inventory = inventory
        is_review_period = ((t - 1) % review_period == 0)

        order_qty = 0.0

        # Review đầu kỳ; do L=0 nên nhận ngay
        if is_review_period:
            inventory_position = inventory
            if inventory_position < order_up_to_level:
                order_qty = order_up_to_level - inventory_position
                inventory += order_qty
                total_order_qty += order_qty
                num_orders += 1
                if order_qty > 0:
                    total_order_cost += order_cost

        inventory_after_replenishment = inventory

        # Nhu cầu xảy ra trong kỳ
        sales = min(inventory, demand)
        shortage = max(demand - inventory, 0.0)
        ending_inventory = max(inventory - demand, 0.0)

        # Chi phí
        period_holding_cost = ending_inventory * holding_cost
        period_shortage_cost = shortage * shortage_cost

        # Cập nhật tổng
        total_sales += sales
        total_shortage_units += shortage
        total_holding_cost += period_holding_cost
        total_shortage_cost += period_shortage_cost

        # Tồn kho chuyển sang kỳ sau
        inventory = ending_inventory

        if return_details:
            records.append({
                "period": t,
                "demand": demand,
                "beginning_inventory": beginning_inventory,
                "review_period_flag": is_review_period,
                "z_value": z,
                "safety_stock": safety_stock,
                "order_up_to_level": order_up_to_level,
                "order_qty": order_qty,
                "inventory_after_replenishment": inventory_after_replenishment,
                "sales": sales,
                "shortage": shortage,
                "ending_inventory": ending_inventory,
                "holding_cost": period_holding_cost,
                "shortage_cost": period_shortage_cost
            })

    #Xuất ra dataframe kết quả
    results_df = pd.DataFrame(records) if return_details else None

    # =========================
    # 3. TỔNG HỢP KẾT QUẢ
    # =========================
    total_demand = demand_series.sum()
    fill_rate = total_sales / total_demand if total_demand > 0 else 1.0
    daily_service_level_empirical = (results_df["shortage"] == 0).mean()

    # Cycle service level thực nghiệm:
    # Một cycle là từ 1 kỳ review đến trước kỳ review tiếp theo
    stockout_cycles = 0
    total_cycles = 0

    for start_idx in range(0, len(demand_series), review_period):
        end_idx = min(start_idx + review_period, len(demand_series))
        cycle_df = pd.DataFrame(records[start_idx:end_idx]) if return_details else None

        if return_details:
            total_cycles += 1
            if cycle_df["shortage"].sum() > 0:
                stockout_cycles += 1

    cycle_service_level_empirical = (
        1 - stockout_cycles / total_cycles if total_cycles > 0 else 1.0
    )

    summary = {
        "mu_demand": mu_demand,
        "std_demand": std_demand,
        "review_period": review_period,
        "lead_time": 0,
        "service_level_target": service_level,
        "z_value": z,
        "safety_stock": safety_stock,
        "expected_demand_over_review": expected_demand_over_review,
        "order_up_to_level": order_up_to_level,
        "initial_inventory": initial_inventory,
        "num_periods_simulated": len(demand_series),
        "num_orders": num_orders,
        "total_demand": total_demand,
        "total_sales": total_sales,
        "total_shortage_units": total_shortage_units,
        "fill_rate": fill_rate,
        "cycle_service_level_empirical": cycle_service_level_empirical,
        "daily_service_level_empirical":daily_service_level_empirical,
        "average_ending_inventory": (
            pd.DataFrame(records)["ending_inventory"].mean() if return_details else None
        ),
        "total_order_qty": total_order_qty,
        "total_holding_cost": total_holding_cost,
        "total_shortage_cost": total_shortage_cost,
        "total_order_cost": total_order_cost,
        "total_cost": total_holding_cost + total_shortage_cost + total_order_cost
    }

    
    return results_df, summary