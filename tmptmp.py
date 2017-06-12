# Create test data
os.chdir("T:/RNA/Baltimore/Jason/ad_hoc/mb/layer1_test")
files = glob.glob('*.csv')
dfs = {}

# list input files
for f in files:
    dfs[os.path.splitext(os.path.basename(f))[0]] = pd.read_csv(f)

# build test data set
for c in dfs.keys():
    pred = dfs[c].iloc[:, [1]]
    df_test   = pd.concat([df, pred], axis=1)

df_test = df_test.drop('ID', axis = 1) # this ID doesn't correspond to the training data anyway

# Fix duplicate colnames
def maybe_dedup_names(names):
    names = list(names)  # so we can index
    counts = {}

    for i, col in enumerate(names):
        cur_count = counts.get(col, 0)

        if cur_count > 0:
            names[i] = '%s.%d' % (col, cur_count)

        counts[col] = cur_count + 1

    return names

cols=pd.Series(df_test.columns)
df_test.columns = maybe_dedup_names(names=cols)

