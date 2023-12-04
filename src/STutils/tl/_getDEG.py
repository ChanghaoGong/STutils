from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from matplotlib.gridspec import GridSpec


def getDEG(
    adata: AnnData,
    cluster: str,
    qval_cutoff: float = 0.1,
    mean_expr_cutoff: float = 0.2,
    top_genes: int = 200,
    layer: str = None,
    save: str = "volcano_plot.pdf",
    key_added: str = "rank_genes_groups",
) -> dict:
    """Get DEGs for rank_genes_groups

    :param adata: adata object
    :type adata: AnnData
    :param cluster: cluster tag
    :type cluster: str
    :param qval_cutoff: qvalue cutoff, defaults to 0.1
    :type qval_cutoff: float, optional
    :param mean_expr_cutoff: mean expression cutoff, defaults to 0.2
    :type mean_expr_cutoff: float, optional
    :param top_genes:  number of top genes, defaults to 200
    :type top_genes: int, optional
    :param layer: adata layers, defaults to None
    :type layer: str, optional
    :param save: if or not save volcano plot, defaults to "volcano_plot.pdf"
    :type save: str, optional
    :param key_added: DEG key, defaults to "rank_genes_groups"
    :type key_added: str, optional
    :return: a DEG dict.
    :rtype: dict
    """
    # sc.tl.rank_genes_groups(adata, cluster, method='wilcoxon')
    result = adata.uns[key_added]
    groups = result["names"].dtype.names
    DEG_dict = OrderedDict()
    diff_genes = pd.DataFrame(
        columns=[
            "group",
            "names",
            "logfoldchanges",
            "pvals",
            "pvals_adj",
            "mean_expression",
        ]
    )
    for group in groups:
        df = pd.DataFrame({key: result[key][group] for key in ["names", "logfoldchanges", "pvals", "pvals_adj"]})
        df["group"] = group
        cluster_cells = adata.obs[cluster] == group
        cluster_data = adata[cluster_cells, :]
        if layer:
            mean_expression = np.mean(cluster_data.layers[layer].toarray(), axis=0)
        else:
            mean_expression = np.mean(cluster_data.X.toarray(), axis=0)
        df = pd.merge(
            df,
            pd.Series(mean_expression, index=cluster_data.var_names, name="mean_expression"),
            left_on="names",
            right_index=True,
        )
        # diff_genes = diff_genes.append(df[['group', 'names', 'logfoldchanges', 'pvals','pvals_adj','mean_expression']])
        diff_genes = pd.concat(
            [
                diff_genes,
                df[
                    [
                        "group",
                        "names",
                        "logfoldchanges",
                        "pvals",
                        "pvals_adj",
                        "mean_expression",
                    ]
                ],
            ],
            axis=0,
        )
        df_filtered = df[
            (df["pvals_adj"] < qval_cutoff) & (df["mean_expression"] > mean_expr_cutoff) & (df["logfoldchanges"] > 0)
        ]
        df_filtered.set_index("names", inplace=True, drop=False)
        # sort rows based on logfoldchange
        df_filtered = df_filtered.iloc[df_filtered["logfoldchanges"].argsort()[::-1]]
        DEG_dict[group] = df_filtered.iloc[0:top_genes, :].index.tolist()
    # 绘制火山图
    ncol = 4
    nrow = len(groups) // 4 + 1
    fig = plt.figure(figsize=(ncol * 6, nrow * 6))
    gs = GridSpec(nrow, ncol, figure=fig)  # 将整个画布分成len(groups)列
    for i, group in enumerate(groups):
        rownum = i // 4
        colnum = i - ((i // 4) * 4)
        ax = fig.add_subplot(gs[rownum, colnum])
        data = diff_genes[diff_genes["group"] == group]
        sns.scatterplot(data=data, x="logfoldchanges", y="mean_expression", color="black")
        sig_genes = data.loc[(data["pvals_adj"] < qval_cutoff) & (data["mean_expression"] > mean_expr_cutoff)]
        sig_genes_top5 = sig_genes.nlargest(5, "logfoldchanges")
        texts = []
        for _, row in sig_genes_top5.iterrows():
            if row["logfoldchanges"] > 0:
                t = ax.text(
                    row["logfoldchanges"],
                    row["mean_expression"],
                    row["names"],
                    ha="right",
                    va="top",
                    color="r",
                )
                texts.append(t)
        # adjust_text(texts, ax=ax)
        ax.set_title(group)
        ax.set_xlabel("Log2FC")
        ax.set_ylabel("mean_expression")

    plt.suptitle("Volcano plot", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save)
    plt.show()
    return DEG_dict
