#author: zhouchi
#date: 20250721

def fromGTF_to_dataframe(GTF_path, result_path=".", feature_type="all", include_attributes=True,col = None):

    import os
    import pandas as pd
    import gffutils
    import warnings

    # 创建临时数据库路径
    db_path = os.path.join(result_path, "gtf_temp.db")
    
    try:
        # 创建数据库（自动推断层级结构）
        db = gffutils.create_db(
            GTF_path,
            dbfn=db_path,
            force=True,
            keep_order=True,
            merge_strategy="merge",
            disable_infer_genes=True, 
            disable_infer_transcripts=True,
            id_spec={"gene": "gene_id", "transcript": "transcript_id"}
        )
        
        # 确定要提取的特征类型
        if feature_type == "all":
            features = list(db.all_features())
        else:
            features = list(db.features_of_type(feature_type))
        
        # 提取特征数据
        records = []
        for feat in features:
            record = {
                "seqid": feat.seqid,
                "source": feat.source,
                "feature_type": feat.featuretype,
                "start": feat.start,
                "end": feat.end,
                "score": feat.score,
                "strand": feat.strand,
                "frame": feat.frame,
                "attributes": str(feat.attributes)
            }
            
            # 解析属性字段为单独列
            if include_attributes:
                for key, value in feat.attributes.items():
                    # 处理多值属性（如exon_id可能有多个值）
                    clean_key = key.replace("-", "_")  # 避免特殊字符
                    record[clean_key] = value[0] if len(value) == 1 else ";".join(value)
            
            records.append(record)
        
        # 转换为DataFrame
        df = pd.DataFrame(records)
        if col is not None:  # 列过滤逻辑[3,5](@ref)
            # 验证列名有效性
            invalid_cols = [c for c in col if c not in df.columns]
            if invalid_cols:
                warnings.warn(f"忽略无效列名: {invalid_cols}", UserWarning)
            df = df[[c for c in col if c in df.columns]]
        # 保存结果
        csv_path = os.path.join(result_path, "gtf_full_features.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ 成功提取 {len(df)} 条特征 → {csv_path}")
        
        return df
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        raise
    finally:
        # 确保数据库连接关闭
        if 'db' in locals():
            db.conn.close()
        
        # 清理临时数据库
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
                print(f"♻️ 临时数据库已清理: {db_path}")
        except Exception as e:
            print(f"⚠️ 数据库清理失败: {e}")


def GeneName_Trans(df, oldName, map_df,
                   Genefrom="gene_id", Geneto="gene_name",
                   Delete_duplicates=False,
                   outname=None,
                   dup_outname=None):  # <--- 新增参数：专门用于指定重复文件的保存路径
    """
    基因名转换 + 可选重复基因处理
    
    参数说明:
    - outname: 转换后的主数据保存路径。如果为 None，则不保存主数据。
    - dup_outname: 被删除的重复项详细信息保存路径。如果为 None，则不保存重复项信息。
    """

    # -------- 读取对照表 --------
    import pandas as pd
    import os
    if isinstance(map_df, str):
        # 增加读取容错，防止路径错误
        if not os.path.exists(map_df):
            raise FileNotFoundError(f"映射表文件不存在: {map_df}")
        map_df = pd.read_csv(map_df)
    elif not isinstance(map_df, pd.DataFrame):
        raise ValueError("map_df must be a DataFrame or a CSV file path.")

    # -------- 检查列 --------
    for col in [Genefrom, Geneto]:
        if col not in map_df.columns:
            raise ValueError(f"Mapping file must contain column: {col}")

    if oldName not in df.columns:
        # 如果找不到列，直接返回原df，防止报错中断整个流程（或者您可以选择 raise Error）
        print(f"[Warning] 列名 {oldName} 在数据中不存在，跳过转换。")
        return df

    # -------- 去重映射 --------
    map_df = map_df[[Genefrom, Geneto]].drop_duplicates(subset=[Genefrom])

    # -------- 构建映射字典 --------
    mapping = dict(zip(map_df[Genefrom], map_df[Geneto]))

    # -------- 应用映射 --------
    # 注意：这里如果没匹配到，保持原值
    df[oldName] = df[oldName].apply(
        lambda x: mapping[x] if x in mapping and pd.notna(mapping[x]) else x
    )

    # ==========================================================
    #           ★ 删除重复基因并导出重复项明细 ★
    # ==========================================================
    if Delete_duplicates:
        # 找到重复基因
        dup_mask = df[oldName].duplicated(keep=False)
        
        # 如果存在重复项
        if dup_mask.any():
            dup_df = df[dup_mask].copy()
            dup_df = dup_df.sort_values(by=oldName)
            
            # 计算总和用于选出最大行
            # 注意：这里假设从第2列开始是数值列。如果数据结构不同，需调整 iloc[:, 1:]
            try:
                dup_df["SumValue"] = dup_df.iloc[:, 1:].sum(axis=1, numeric_only=True)
            except Exception:
                dup_df["SumValue"] = 0 # 如果无法计算，设为0
            
            # -------- 【新功能】保存重复项明细 --------
            if dup_outname is not None:
                # 确保目录存在
                dup_dir = os.path.dirname(dup_outname)
                if dup_dir and not os.path.exists(dup_dir):
                    os.makedirs(dup_dir)
                    
                dup_df.to_csv(dup_outname, index=False)
                # print(f"    -> [Info] 重复项明细已保存: {os.path.basename(dup_outname)}")

            # -------- 针对每个基因保留总和最大的行 --------
            keep_rows = []
            for gene, subdf in dup_df.groupby(oldName):
                idx_max = subdf["SumValue"].idxmax()
                keep_rows.append(idx_max)

            # 在原 df 中保留这些行，删除其他重复行
            df = df.loc[~dup_mask | df.index.isin(keep_rows)]

    # ==========================================================
    #                 ★ 写出最终结果 ★
    # ==========================================================
    # 只有当 outname 被明确指定（不是None也不是空字符串）时才保存
    if outname:
        df.to_csv(f"{outname}.csv", index=False)
        print(f"    -> 结果已保存至: {outname}.csv")

    return df