import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# --- 全局绘图样式设置 ---
sns.set_theme(style="whitegrid")
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['pdf.fonttype'] = 42 # 确保PDF中的字体可编辑

def process_KEGG_category(KEGGfile, pcut=0.05, padjcut=1, qcut=1,
                          cat='category', specified=None, n_cat=5,
                          n_pathway=10, credence="pvalue", outdir="KEGG_barplot_by_category"):
    """
    处理KEGG富集分析结果，按类别生成条形图。
    """
    # --- 1. 初始化和参数检查 ---
    print("🚀 开始处理KEGG富集结果...")
    result_dir = "./" + outdir
    barplot_dir = os.path.join(result_dir, "Barresult")
    os.makedirs(barplot_dir, exist_ok=True)
    print(f"✅ 已创建输出文件夹：{os.path.abspath(result_dir)}")

    valid_cats = ['category', 'subcategory']
    if cat not in valid_cats:
        raise ValueError(f"参数 'cat' 只能是 {valid_cats} 之一。你输入的是: '{cat}'")
    valid_credence = ['pvalue', 'p.adjust', "qvalue"]
    if credence not in valid_credence:
        raise ValueError(f"参数 'credence' 只能是 {valid_credence} 之一。你输入的是: '{credence}'")
    print("✅ 参数合法性检查通过")

    # --- 2. 数据读取和过滤 ---
    try:
        df = pd.read_csv(KEGGfile)
        print(f"📂 成功读取数据文件：{KEGGfile}，原始数据共 {len(df)} 条通路")
    except Exception as e:
        raise IOError(f"❌ 无法读取文件 {KEGGfile}，错误信息：{e}")

    df_filtered = df[(df['pvalue'] <= pcut) & (df['p.adjust'] <= padjcut) & (df['qvalue'] <= qcut)]
    print(f"🔍 根据阈值过滤完成，剩余 {len(df_filtered)} 条通路")

    if df_filtered.empty:
        print("⚠️ 过滤后无数据，程序终止。")
        return
    if cat not in df_filtered.columns:
        raise KeyError(f"❌ 数据中不包含字段 '{cat}'，请检查输入文件列名是否正确。")

    # --- 3. 按分类统计和处理 ---
    cat_counts = df_filtered[cat].value_counts().reset_index()
    cat_counts.columns = [cat, 'frequency']
    cat_counts = cat_counts.sort_values(by='frequency', ascending=False).reset_index(drop=True)
    catlist_path = os.path.join(result_dir, "1_CategoryToplist.txt")
    cat_counts.to_csv(catlist_path, sep="\t", index=False)
    print(f"📊 分类统计完成，已保存至：{catlist_path}")

    # --- 内部绘图函数 ---
    def plot_and_save(df_to_plot, title_prefix, output_name):
        print(f"  📈 正在为 '{title_prefix}' 绘制条形图...")
        df_to_plot['value'] = -np.log10(df_to_plot[credence])
        
        if len(df_to_plot) > 1:
            normed_vals = (df_to_plot['value'] - df_to_plot['value'].min()) / (df_to_plot['value'].max() - df_to_plot['value'].min() + 1e-9)
        else:
            normed_vals = [0.5]
        
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#6ab3cd', '#f8dfa9', '#d26e75'])
        colors = custom_cmap(normed_vals)
        
        # 单个图的高度也根据条目数动态调整
        fig_height = 1.5 + len(df_to_plot) * 0.6
        fig, ax = plt.subplots(figsize=(8, fig_height))
        
        ax.barh(df_to_plot['Description'], df_to_plot['value'], color=colors)
        ax.set_xlabel(f"-log10({credence})")
        ax.set_ylabel("Pathway")
        ax.invert_yaxis()
        ax.grid(False)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
        
        fig.suptitle(f"{title_prefix} - Top {len(df_to_plot)} Pathways", fontweight='bold', fontsize=10)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        pdf_path = os.path.join(barplot_dir, f"{output_name}_barplot.pdf")
        fig.savefig(pdf_path)
        plt.close(fig)
        print(f"  ✅ 图像已保存至：{pdf_path}")

    # --- 4. 循环处理每个分类并绘图 ---
    subplot_data = []
    if specified:
        categories_to_process = [specified] if isinstance(specified, str) else specified
    else:
        print(f"🔄 未指定分类，将自动处理Top {n_cat}个分类...")
        categories_to_process = cat_counts[cat].tolist()[:int(n_cat)]

    for category_name in categories_to_process:
        print(f"\n📌 开始处理分类：'{category_name}'")
        df_sub = df_filtered[df_filtered[cat] == category_name]
        df_tobar = df_sub.sort_values(by=credence, ascending=True).head(n_pathway).copy()

        if df_tobar.empty:
            print(f"  ⚠️ 分类 '{category_name}' 中无数据可供绘图，已跳过。")
            continue
            
        out_path = os.path.join(result_dir, f"2_{category_name}_path.txt")
        df_tobar.to_csv(out_path, sep="\t", index=False)
        print(f"  ✍️ 通路数据已保存至：{out_path}")
        
        plot_and_save(df_tobar, category_name, "_".join(category_name.split()))
        subplot_data.append((category_name, df_tobar))

    # --- 5. 拼接所有分类的图 (*** MODIFIED SECTION ***) ---
    if len(subplot_data) > 0:
        print("\n🧩 正在拼接所有分类的条形图...")
        n_rows = len(subplot_data)

        # 1. 计算每个子图的条目数，这将作为高度分配的比例
        height_ratios = [len(df_plot) for _, df_plot in subplot_data]
        total_bars = sum(height_ratios)
        
        # 2. 根据总条目数和子图数量，动态计算出最合适的总高度
        # 每个条目大约占0.6英寸，每个子图的额外空间（标题、坐标轴、间距）大约占1.5英寸
        total_height = max(n_rows * 2, total_bars * 0.6 + n_rows * 1.5)
        
        # 3. 使用 gridspec_kw 来指定子图的高度按比例分配
        fig, axs = plt.subplots(n_rows, 1, 
                                figsize=(9, total_height), 
                                squeeze=False,
                                gridspec_kw={'height_ratios': height_ratios})
        
        axs = axs.flatten()

        for ax, (label, df_plot) in zip(axs, subplot_data):
            value_col = -np.log10(df_plot[credence])
            if len(df_plot) > 1:
                normed_vals = (value_col - value_col.min()) / (value_col.max() - value_col.min() + 1e-9)
            else:
                normed_vals = [0.5]
            
            custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#6ab3cd', '#f8dfa9', '#d26e75'])
            colors = custom_cmap(normed_vals)
            
            ax.barh(df_plot['Description'], value_col, color=colors)
            ax.set_title(f"{label} - Top {len(df_plot)} Pathways")
            ax.set_xlabel(f"-log10({credence})")
            # ax.set_ylabel("Pathway") # 在拼接图中，Y轴标签可以省略以节省空间
            ax.invert_yaxis()
            ax.grid(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # 使用 pad 参数来增加子图之间的垂直间距
        fig.tight_layout(pad=3.0)
        all_fig_path = os.path.join(barplot_dir, "AllCategories_barplot.pdf")
        fig.savefig(all_fig_path)
        plt.close(fig)
        print(f"✅ 所有分类拼接图已保存至：{all_fig_path}")
        
    print("🎉 KEGG处理完成！")

def process_GO_category(GOfile, pcut=0.05, padjcut=1, qcut=1,
                        n_pathway=10, credence="pvalue", outdir="GO_barplot"):
    """
    处理GO富集分析结果，按BP, CC, MF三类分别生成条形图。
    """
    # --- 1. 初始化和参数检查 ---
    print("\n🚀 开始处理GO富集结果...")
    os.makedirs(outdir, exist_ok=True)
    barplot_dir = os.path.join(outdir, "Barresult")
    os.makedirs(barplot_dir, exist_ok=True)
    print(f"✅ 已创建输出文件夹：{os.path.abspath(outdir)}")

    valid_credence = ['pvalue', 'p.adjust', 'qvalue']
    if credence not in valid_credence:
        raise ValueError(f"❌ 'credence' 参数值无效，必须是 {valid_credence} 之一。")
    print("✅ 参数合法性检查通过")

    # --- 2. 数据读取 ---
    try:
        df = pd.read_csv(GOfile)
        print(f"📂 成功读取GO文件: {GOfile}，共 {len(df)} 条记录")
    except Exception as e:
        raise IOError(f"❌ 无法读取文件 {GOfile}，错误信息：{e}")

    # --- 内部处理和绘图函数 ---
    def process_and_plot_ontology(df_ontology, category_name):
        print(f"\n📌 开始处理类别：{category_name} (原始记录数: {len(df_ontology)})")

        df_tobar = df_ontology[(df_ontology['pvalue'] <= pcut) &
                               (df_ontology['p.adjust'] <= padjcut) &
                               (df_ontology['qvalue'] <= qcut)].copy()
        print(f"  🔍 过滤后剩余 {len(df_tobar)} 条记录")

        if df_tobar.empty:
            print(f"  ⚠️ 过滤后无数据，跳过绘图：{category_name}")
            return

        df_tobar = df_tobar.sort_values(by=credence, ascending=True).head(n_pathway)
        print(f"  📊 选取前 {len(df_tobar)} 条路径用于绘图")

        txt_path = os.path.join(outdir, f"{category_name}_top_{len(df_tobar)}_pathways.txt")
        df_tobar.to_csv(txt_path, sep='\t', index=False)
        print(f"  ✍️ 结果保存至：{txt_path}")

        df_tobar['value'] = -np.log10(df_tobar[credence])
        
        # 创建颜色映射
        if len(df_tobar) > 1:
            normed_vals = (df_tobar['value'] - df_tobar['value'].min()) / (df_tobar['value'].max() - df_tobar['value'].min() + 1e-9)
        else:
            normed_vals = [0.5]
            
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#6ab3cd', '#f8dfa9', '#d26e75'])
        colors = custom_cmap(normed_vals)

        # --- 绘图逻辑 ---
        fig, ax = plt.subplots(figsize=(8, 1+len(df_tobar) * 0.6))
        
        ax.barh(df_tobar['Description'], df_tobar['value'], color=colors)
        ax.set_xlabel(f"-log10({credence})")
        ax.set_ylabel("Pathway")
        ax.invert_yaxis()
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig.suptitle(f"{category_name} - Top {len(df_tobar)} Pathways", fontweight='normal', fontsize=10)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        fig_path = os.path.join(barplot_dir, f"{category_name}_barplot.pdf")
        fig.savefig(fig_path)
        plt.close(fig)
        print(f"  📈 图像保存至：{fig_path}")

    # --- 3. 分别处理 BP / CC / MF ---
    ontologies = {'BP': 'Biological Process', 'CC': 'Cellular Component', 'MF': 'Molecular Function'}
    for short_name, long_name in ontologies.items():
        if 'ONTOLOGY' in df.columns:
            df_sub = df[df['ONTOLOGY'] == short_name]
            if not df_sub.empty:
                 process_and_plot_ontology(df_sub, short_name)
            else:
                 print(f"\n⚠️ 在数据中未找到 '{short_name}' ({long_name}) 的条目，已跳过。")
        else:
            raise KeyError("❌ 输入文件中未找到 'ONTOLOGY' 列，无法区分BP, CC, MF。")
            
    print("\n🎉 所有 GO 类别处理完成！")


#整合分析
def from_foldfile_to_enrich_analysis(
    indir, 
    go_suffix=".All.GO.list.txt", 
    kegg_suffix=".All.KEGG.list.to.symbl.txt",
    outdir="Enrich_Results",
    # 通用参数
    pcut=0.05, padjcut=1, qcut=1, 
    n_cat=5, n_pathway=10, credence="pvalue", cat="category", specified=None
):
    """
    批量处理当前文件夹下的GO和KEGG富集结果文件。
    1. 筛选指定后缀的文件；
    2. 去掉后缀作为file名；
    3. 在outdir下为每个file创建子目录；
    4. 调用process_GO_category / process_KEGG_category绘图。
    """

    import os
    import sys
    sys.path.append("/BDproject1/User/zhouchi/tools/zc_python_module")
    from Enrich_analysis_barplot_by_category_v1 import process_KEGG_category, process_GO_category

    # 确保输出目录存在
    os.makedirs(outdir, exist_ok=True)

    # 遍历当前目录，找出目标文件
    files = os.listdir(indir)
    go_files = [f for f in files if f.endswith(go_suffix)]
    kegg_files = [f for f in files if f.endswith(kegg_suffix)]

    print(f"🔍 发现 {len(go_files)} 个GO文件，{len(kegg_files)} 个KEGG文件")

    # 处理GO文件
    for gofile in go_files:
        file_prefix = gofile.replace(go_suffix, "")
        file_outdir = os.path.join(outdir, file_prefix)
        os.makedirs(file_outdir, exist_ok=True)
        print(f"\n🚀 开始处理GO文件: {gofile} -> 输出目录 {file_outdir}")
        
        process_GO_category(
            GOfile=os.path.join(indir, gofile),
            pcut=pcut, padjcut=padjcut, qcut=qcut,
            n_pathway=n_pathway, credence=credence,
            outdir=file_outdir
        )

    # 处理KEGG文件
    for keggfile in kegg_files:
        file_prefix = keggfile.replace(kegg_suffix, "")
        file_outdir = os.path.join(outdir, file_prefix)
        os.makedirs(file_outdir, exist_ok=True)
        print(f"\n🚀 开始处理KEGG文件: {keggfile} -> 输出目录 {file_outdir}")

        process_KEGG_category(
            KEGGfile=os.path.join(indir, keggfile),
            pcut=pcut, padjcut=padjcut, qcut=qcut,
            cat=cat, specified=specified, n_cat=n_cat,
            n_pathway=n_pathway, credence=credence,
            outdir=file_outdir
        )

    print("\n🎉 所有文件处理完成！")