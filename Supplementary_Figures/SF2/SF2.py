# %%
import pandas as pd
import matplotlib.pyplot as plt
from upsetplot import UpSet, from_memberships
import matplotlib.patches as mpatches
import seaborn as sns


def load_metadata(csv_file_path, valid_ids_path):
    metadata_df = pd.read_csv(csv_file_path)
    with open(valid_ids_path, "r") as f:
        valid_ids = {line.strip() for line in f}
    metadata_df['ID'] = metadata_df['ID'].str.replace('-', '_', regex=False)
    return metadata_df[metadata_df["ID"].astype(str).isin(valid_ids)]


def create_upset_plot(metadata_df, out_path, title, sort_by='input', sort_categories_by='-cardinality'):
    metadata_df = metadata_df[["ID", "Treatment Response", "Type of Visit", "Patient Number"]]
    patient_visits = pd.DataFrame({
        'Screen': metadata_df.groupby('Patient Number')['Type of Visit'].apply(lambda x: 'Screen' in x.values),
        'Day 0': metadata_df.groupby('Patient Number')['Type of Visit'].apply(lambda x: 'Day 0' in x.values),
        'Adj Wk 1': metadata_df.groupby('Patient Number')['Type of Visit'].apply(lambda x: 'Adj Wk 1' in x.values)
    })
    memberships = [
        [visit for visit, present in row.items() if present] for _, row in patient_visits.iterrows()
    ]
    upset_data = from_memberships(memberships)
    upset = UpSet(upset_data, subset_size='count', show_counts='%d', facecolor="lightpink", sort_by=sort_by,
                  sort_categories_by=sort_categories_by)
    upset.plot()
    plt.title(title, y=1.05)
    plt.savefig(out_path, dpi=1000, bbox_inches='tight')
    plt.show()


def create_institute_barplot(metadata_df, out_path, title, cohort=None, y_ticks=range(0, 31, 5)):
    universities = [
        "University of Cincinnati", "University of Michigan", "Ohio State University", 
        "Medical University of South Carolina", "MD Anderson", "University of Louisville"
    ]
    institute_label_map = {
        "University of Cincinnati": "UC", "University of Michigan": "U-M", "MD Anderson": "MD Anderson",
        "University of Louisville": "UofL", "Ohio State University": "OSU",
        "Medical University of South Carolina": "MUSC"
    }
    metadata_df = metadata_df[metadata_df["Institute"].isin(universities)]
    metadata_df["Institute"] = pd.Categorical(metadata_df["Institute"], categories=universities, ordered=True)
    if cohort:
        metadata_df["Cohort"] = metadata_df["Institute"].apply(lambda x: "Testing" if x in cohort else "Training")
    visit_counts = metadata_df.groupby(["Institute", "Type of Visit"]).size().reset_index(name='Count')
    pivot_df = visit_counts.pivot(index="Institute", columns="Type of Visit", values="Count").fillna(0)
    pivot_df = pivot_df.reindex(universities)

    colorblind_palette = sns.color_palette("colorblind")
    is_response_stratified = set(pivot_df.columns) == {"Responder", "Non-Responder"}
    if is_response_stratified:
        colors = {
            "Responder": colorblind_palette[0],
            "Non-Responder": colorblind_palette[5],
        }
        pivot_df = pivot_df[["Responder", "Non-Responder"]]
    else:
        colors = ["lightblue", "lightgreen", "lightcoral"]

    fig, ax = plt.subplots(figsize=(4, 4))
    if cohort:
        for idx in [universities.index(inst) for inst in cohort]:
            ax.axvspan(idx - 0.5, idx + 0.5, facecolor='lightgray', alpha=0.5)

    pivot_df.plot(kind='bar', stacked=False,
                  color=colors if isinstance(colors, list) else [colors[col] for col in pivot_df.columns], ax=ax)

    plt.ylabel('Number of Samples')
    ax.set_xticklabels([institute_label_map.get(label.get_text(), label.get_text()) for label in ax.get_xticklabels()],
                       rotation=30, ha='right')
    plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.5)
    ax.set_yticks(y_ticks)
    if cohort:
        patch = mpatches.Patch(color='lightgray', alpha=0.5, label='Institute Hold-Out')
        handles, labels = ax.get_legend_handles_labels()
        handles.append(patch)
        labels.append('Institute Hold-Out')
        ax.legend(handles=handles, labels=labels, title=None, loc='upper right')
    else:
        plt.legend(loc='upper right')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=1000, bbox_inches='tight')
    plt.show()


def create_date_barplot(metadata_df, date_column, out_path, title, custom_order, y_ticks):
    visit_counts = metadata_df.groupby([date_column, "Type of Visit"]).size().reset_index(name='Count')
    pivot_df = visit_counts.pivot(index=date_column, columns="Type of Visit", values="Count").fillna(0)
    pivot_df = pivot_df.reindex(custom_order, fill_value=0)
    pivot_df.index = pd.Categorical(pivot_df.index, categories=custom_order, ordered=True)
    pivot_df.sort_index(inplace=True)

    colorblind_palette = sns.color_palette("colorblind")
    is_response_stratified = set(pivot_df.columns) == {"Responder", "Non-Responder"}
    if is_response_stratified:
        colors = {
            "Responder": colorblind_palette[0],
            "Non-Responder": colorblind_palette[5],
        }
        pivot_df = pivot_df[["Responder", "Non-Responder"]]
    else:
        colors = ["lightblue", "lightgreen", "lightcoral"]

    fig, ax = plt.subplots(figsize=(4, 4))
    pivot_df.plot(kind='bar', stacked=False,
                  color=colors if isinstance(colors, list) else [colors[col] for col in pivot_df.columns], ax=ax)

    plt.ylabel('Number of Samples')
    plt.xlabel(date_column)
    plt.legend(loc='upper right')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.5)
    ax.set_yticks(y_ticks)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=1000, bbox_inches='tight')
    plt.show()


# Paths
csv_file = "../../Supplementary_Tables/ST1/RAW_HNSCC_METADATA_NEW_v10.csv"
cv_ids = "../../Utils/Lists/cv_ids.txt"
holdout_ids = "../../Utils/Lists/holdout_ids.txt"

# Load metadata
metadata_cv = load_metadata(csv_file, cv_ids)
metadata_holdout = load_metadata(csv_file, holdout_ids)

# SF2.P1.v1.pdf
create_upset_plot(
    metadata_cv,
    out_path='SF2.P1.v1.pdf',
    title='Analysis Set',
    sort_by='input',
    sort_categories_by='-cardinality'
)

# SF2.P2.v1.pdf
create_upset_plot(
    metadata_holdout,
    out_path='SF2.P2.v1.pdf',
    title='Pre-Holdout Set',
    sort_by='-degree',
    sort_categories_by='input'
)

# SF2.P3a.v1.pdf — stratified by Type of Visit (default)
create_institute_barplot(
    metadata_cv,
    out_path='SF2.P3a.v1.pdf',
    title='Analysis Set (by Visit Type)',
    #cohort={"Ohio State University", "University of Louisville"},
    y_ticks=range(0, 31, 5)
)

# SF2.P3b.v1.pdf — stratified by Treatment Response
create_institute_barplot(
    metadata_cv.drop(columns=["Type of Visit"]).rename(columns={"Treatment Response": "Type of Visit"}),
    out_path='SF2.P3b.v1.pdf',
    title='Analysis Set (by Treatment Response)',
    #cohort={"Ohio State University", "University of Louisville"},
    y_ticks=range(0, 51, 5)
)

# SF2.P4a.v1.pdf - Pre-Holdout by Visit
create_institute_barplot(
    metadata_holdout,
    'SF2.P4a.v1.pdf',
    'Pre-Holdout Set (by Visit Type)',
    None,
    range(0, 6, 1)
)

# SF2.P4b.v1.pdf - Pre-Holdout by Treatment Response
create_institute_barplot(
    metadata_holdout.drop(columns=["Type of Visit"]).rename(columns={"Treatment Response": "Type of Visit"}),
    'SF2.P4b.v1.pdf',
    'Pre-Holdout Set (by Treatment Response)',
    None,
    range(0, 11, 1)
)


# SF2.P5a.v1.pdf - Analysis Set cfDNA Isolation Date by Visit
create_date_barplot(
    metadata_cv,
    "cfDNA Isolation Date",
    "SF2.P5a.v1.pdf",
    "Analysis Set (by Visit Type)",
    custom_order=["12/26/19", "12/30/19", "12/31/19", "1/2/20", "1/3/20", "1/15/20", "9/16/20", "9/17/20", "9/18/20"],
    y_ticks=range(0, 15, 1)
)

# SF2.P5b.v1.pdf - by Treatment Response
create_date_barplot(
    metadata_cv.drop(columns=["Type of Visit"]).rename(columns={"Treatment Response": "Type of Visit"}),
    "cfDNA Isolation Date",
    "SF2.P5b.v1.pdf",
    "Analysis Set (by Treatment Response)",
    custom_order=["12/26/19", "12/30/19", "12/31/19", "1/2/20", "1/3/20", "1/15/20", "9/16/20", "9/17/20", "9/18/20"],
    y_ticks=range(0, 25, 5)
)


# SF2.P6a.v1.pdf - Pre-Holdout cfDNA Date by Visit
create_date_barplot(
    metadata_holdout,
    "cfDNA Isolation Date",
    "SF2.P6a.v1.pdf",
    "Pre-Holdout Set (by Visit Type)",
    custom_order=["12/26/19", "12/30/19", "12/31/19", "1/2/20", "1/3/20", "1/15/20", "9/16/20", "9/17/20", "9/18/20"],
    y_ticks=range(0, 5, 1)
)

# SF2.P6b.v1.pdf - Pre-Holdout cfDNA Date by Response
create_date_barplot(
    metadata_holdout.drop(columns=["Type of Visit"]).rename(columns={"Treatment Response": "Type of Visit"}),
    "cfDNA Isolation Date",
    "SF2.P6b.v1.pdf",
    "Pre-Holdout Set (by Treatment Response)",
    custom_order=["12/26/19", "12/30/19", "12/31/19", "1/2/20", "1/3/20", "1/15/20", "9/16/20", "9/17/20", "9/18/20"],
    y_ticks=range(0, 10, 1)
)


# SF2.P7a.v1.pdf - WGS Prep by Visit
create_date_barplot(
    metadata_cv,
    "WGS Library Prep Date",
    "SF2.P7a.v1.pdf",
    "Analysis Set (by Visit Type)",
    custom_order=["12/14/20", "12/15/20", "2/26/20", "2/27/20", "2/28/20", "3/1/20", "3/8/20", "3/11/20"],
    y_ticks=range(0, 15, 5)
)

# SF2.P7b.v1.pdf - WGS Prep by Response
create_date_barplot(
    metadata_cv.drop(columns=["Type of Visit"]).rename(columns={"Treatment Response": "Type of Visit"}),
    "WGS Library Prep Date",
    "SF2.P7b.v1.pdf",
    "Analysis Set (by Treatment Response)",
    custom_order=["12/14/20", "12/15/20", "2/26/20", "2/27/20", "2/28/20", "3/1/20", "3/8/20", "3/11/20"],
    y_ticks=range(0, 26, 5)
)


# SF2.P8a.v1.pdf - Holdout WGS Prep by Visit
create_date_barplot(
    metadata_holdout,
    "WGS Library Prep Date",
    "SF2.P8a.v1.pdf",
    "Pre-Holdout Set (by Visit Type)",
    custom_order=["12/14/20", "12/15/20", "2/26/20", "2/27/20", "2/28/20", "3/1/20", "3/8/20", "3/11/20"],
    y_ticks=range(0, 5, 1)
)

# SF2.P8b.v1.pdf - Holdout WGS Prep by Response
create_date_barplot(
    metadata_holdout.drop(columns=["Type of Visit"]).rename(columns={"Treatment Response": "Type of Visit"}),
    "WGS Library Prep Date",
    "SF2.P8b.v1.pdf",
    "Pre-Holdout Set (by Treatment Response)",
    custom_order=["12/14/20", "12/15/20", "2/26/20", "2/27/20", "2/28/20", "3/1/20", "3/8/20", "3/11/20"],
    y_ticks=range(0, 8, 1)
)
