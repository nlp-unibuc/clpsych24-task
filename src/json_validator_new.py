import pandas as pd
import os
import json


all_user_ids_to_post_ids = {
    "19041": ["1ftvgt", "2j2t7i"],
    "19239": ["1wj80m"],
    "19788": ["36eh73", "3am642"],
    "48665": ["ev6tp"],
    "1523": ["3ejsiv"],
    "6939": ["17r00i", "3fuxue"],
    "6998": ["3c83gc"],
    "7302": ["3gcaik"],
    "10090": ["22l8w4", "2mb3ku"],
    "11388": ["2m9gm4"],
    "12559": ["2sq2me"],
    "14958": ["1shwnv"],
    "15807": ["l5jxp"],
    "18608": ["2v0w14"],
    "18639": ["2f7wj4"],
    "20277": ["3hy1hk", "3i9p8b"],
    "21653": ["16xkll"],
    "23080": ["1gceob", "1t7se3"],
    "23341": ["3dthkh"],
    "23400": ["1y298o"],
    "24003": ["2jo2q0"],
    "24331": ["2zienb"],
    "27262": ["2u5z30"],
    "29356": ["26nqsc"],
    "30179": ["2evvoo"],
    "34032": ["1otrx6"],
    "34464": ["35lq71"],
    "34502": ["12umtk"],
    "37770": ["3f9nd4", "3gk55c"],
    "38531": ["2qlffz"],
    "39656": ["30reha"],
    "39832": ["1xpagm", "2017wz"],
    "40671": ["357o7f"],
    "41651": ["2scmvs", "2u91iq"],
    "43030": ["2fwbqi"],
    "43328": ["2x08xn"],
    "44826": ["1k3agb"],
    "46267": ["2nmehc"],
    "49038": ["2gt6wk", "2hnma9"],
    "11360": ["2xrnry"],
    "13321": ["3ic4t8"],
    "14075": ["32b7zk", "33kh5m", "35ylua"],
    "16852": ["1631gw"],
    "16867": ["30xti7", "32jo2k"],
    "18321": ["21mm33", "241gpm"],
    "19250": ["3gizbj"],
    "21926": ["1ph5yt"],
    "28817": ["36xc3s", "383n1f", "3cq8kt"],
    "29142": ["38bc2o"],
    "31972": ["2iz5y7"],
    "33368": ["2f27hy"],
    "33824": ["33r8b6"],
    "34268": ["2z2q1f"],
    "37759": ["3152g6"],
    "46616": ["10llu1"],
    "46771": ["3896jf"],
    "24231": ["23tvqs"],
    "35822": ["24u2gv"],
    "44554": ["2jfo9n", "2onplv"],
    "47912": ["thn6n"],
    "3434": ["3h0z05"],
    "5916": ["280apu"],
    "8368": ["1azxoz"],
    "8547": ["39akaq"],
    "10940": ["2h4b4q"],
    "13753": ["2vcol2"],
    "14272": ["yaqxi"],
    "14302": ["2b3y50", "2booi6"],
    "14680": ["16d5de"],
    "15722": ["31dz04"],
    "16366": ["3dv1gk"],
    "16596": ["2rp5bb", "319ioh"],
    "16673": ["2uquru"],
    "17606": ["1tplbc"],
    "18616": ["xfyos"],
    "19510": ["3gx2sj"],
    "25320": ["2aqb2a", "2kvmof"],
    "27214": ["mxci8", "13c9hl", "1455fd"],
    "31509": ["2c7y41"],
    "31683": ["3gjkm6"],
    "32490": ["2h1vke"],
    "32714": ["35miyb"],
    "34078": ["2802g5"],
    "36117": ["1mjszi", "2d3de4"],
    "43263": ["25rnwu"],
    "43701": ["3hxe6j"],
    "44656": ["278mzu"],
    "44846": ["2yek5b"],
    "45909": ["2ng6fr"],
    "46258": ["1wb4qr", "2wsqbn"],
    "46530": ["20d48v"],
    "47012": ["37jcax"],
    "47614": ["pnkh4", "15j1ft"],
    "47899": ["19jxq1"],
    "50326": ["3akx6o"],
    "3058": ["3iagg6"],
    "3928": ["32i9nz"],
    "5224": ["2a7wms"],
    "14737": ["3fvr4l"],
    "17820": ["k76ov"],
    "19058": ["2zqq2c", "32o53l"],
    "19916": ["2e5zho"],
    "20007": ["1w6g6u"],
    "29501": ["1getay", "1gwlbb"],
    "30593": ["2uafbk"],
    "32858": ["2y7a6g", "3ci2ni"],
    "38515": ["1pl51l"],
    "42410": ["1zhg6p", "224i1n", "227n97"],
    "47251": ["2dq4gx"],
    "47572": ["2tfdb4", "3ff64x"],
    "50337": ["3f7iqw"],
    "51395": ["189w0w"],
    "123": ["2v6c9q", "3e8si7"],
    "3697": ["2ff664"],
    "5616": ["g7f7g"],
    "6303": ["20kbpj"],
    "6572": ["2xa7jl"],
    "243": ["2j44ow"],
    "777": ["1kpxiu"],
    "836": ["1d7nab", "1unus3"],
    "1270": ["2dv5mw"],
    "373": ["2xe7hp", "3bhds2", "3d2ttx"],
    "899": ["3h9o7o", "3hh29z"],
    "1264": ["vkxfb", "2nledl"],
    "2421": ["2il6xf"],
}


def load_csv(file_path):
    """
    Load a CSV file into a Pandas DataFrame and raise error if file doesn't exist.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - pd.DataFrame: The DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    return pd.read_csv(file_path)


def merge_dataframes(
    df1, df2, on_column, labels=["b", "c", "d"], subreddit="SuicideWatch"
):
    """
    Merge two DataFrames based on a common column.

    Parameters:
    - df1 (pd.DataFrame): The first DataFrame.
    - df2 (pd.DataFrame): The second DataFrame.
    - on_column (str): The column on which to merge the DataFrames.

    Returns:
    - pd.DataFrame: The merged DataFrame.
    """
    merged_df = pd.merge(df1, df2, on=on_column)
    merged_df = merged_df[merged_df["label"].isin(labels)]
    return merged_df[merged_df["subreddit"] == subreddit]


def load_json(file_path):
    """
    Load a JSON file and return the content as a Python dictionary.

    Parameters:
    - file_path (str): Path to the JSON file.

    Returns:
    - dict: The content of the JSON file as a dictionary.
    """
    try:
        with open(file_path, "r") as file:
            json_content = json.load(file)
        return json_content
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file '{file_path}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def check_mapping_of_ids(submission_data):
    """
    Checks if submitted data contains all expected users and if posts are correctly mapped to users.
    """
    expected_users = set([u for u in all_user_ids_to_post_ids.keys()])
    found_users = set([u for u in submission_data.keys()])
    missing_users = expected_users - found_users
    assert (
        not missing_users
    ), f"The following users are missing from your json file: {missing_users}"

    for user in expected_users:
        expected_pids = set(all_user_ids_to_post_ids[user])
        found_pids = set([k["post_id"] for k in submission_data[str(user)]["posts"]])
        missing_pids = expected_pids - found_pids
        assert (
            not missing_pids
        ), f"Missing post_ids for user {user} in your json file. \nExpected:\t{set(expected_pids)}\nFound:\t\t{set(found_pids)}"


def check_json_structure(submission_data, data):
    """
    Checks submitted data for expected fields and data types.
    """
    if not isinstance(submission_data, dict):
        raise TypeError(
            f"Expected submission file to be json loaded into python Dict, got {type(submission_data)}"
        )

    for user, _dict in submission_data.items():
        for key, value in _dict.items():

            if key not in ["summarized_evidence", "posts", "optional"]:
                print(f"Found unexpected key {key} in data for user {user}")

            # Check if summarized_evidence is string
            if key == "summarized_evidence":
                assert isinstance(
                    value, str
                ), f"The summarized evidence is not in a valid (string) format for user {user}."

            # Check if optional either a list or nested list of strings
            if key == "optional" and value:
                for rc in value:
                    assert isinstance(
                        rc, list
                    ), f"Expected list of string lists in reasoning chain submission ('optional' in json), got list of {type(rc)} for user {user}"
                    for step in rc:
                        assert all(
                            isinstance(step, str)
                        ), f"Expected list of string lists in reasoning chain submission ('optional' in json), got lists of {type(step)} lists for user {user}"

            if key == "posts":
                for post in value:
                    post_id = post["post_id"]
                    highlights = post["highlights"]
                    if highlights:
                        assert all(
                            isinstance(h, str) for h in highlights
                        ), f"Highlights are not in a valid list-of-strings format for user {user} post_id {post_id}"
                        assert all(
                            h for h in highlights
                        ), f"Found empty highlights for user {user} post_id {post_id}"
                        # Check if highlights exist within the post content (Note: this is NOT case sensitive)
                        df_post = data[data["post_id"] == post_id]
                        _post_title = str(df_post["post_title"].iloc[0])
                        _post_body = str(df_post["post_body"].iloc[0])
                        content = (_post_title + " " + _post_body).lower().strip()
                        for h in highlights:
                            assert h.lower().strip() in content, (
                                "The Highlighted phrase '%s' does not match the content of post %s for the user %d."
                                % (h.strip(), post_id, int(user))
                            )


def main(csv_expert_path, csv_expert_posts_path, submission_json_path):
    # Load dataset
    loaded_expert = load_csv(file_path=csv_expert_path)
    loaded_expert_posts = load_csv(file_path=csv_expert_posts_path)
    merged_df = merge_dataframes(
        df1=loaded_expert, df2=loaded_expert_posts, on_column="user_id"
    )

    # Load submitted data
    with open(submission_json_path, "r") as file:
        submission_data = json.load(file)

    # Check post count in original dataset
    assert merged_df.shape[0] == 332, (
        f"Found "
        + str(len(merged_df))
        + " SuicideWatch posts by the b/c/d users, but should have found 332. Please check your expert/expert.csv and expert/expert_posts.csv files."
    )

    # Check user count in submitted data
    submitted_user_ids = [int(x) for x in submission_data.keys()]
    num_submitted_user_ids = len(set(submitted_user_ids))
    assert (
        num_submitted_user_ids == 125
    ), f"Expected 125 users, found {num_submitted_user_ids} users in {submission_json_path}."

    # Apply filters + check post count in submitted data
    merged_df = merged_df[merged_df["user_id"].isin(submitted_user_ids)]
    merged_df = merged_df.sort_values(
        by=["user_id", "post_id", "timestamp"]
    ).reset_index(drop=True)
    num_posts_from_submitted_user_ids = merged_df.shape[0]
    assert (
        num_posts_from_submitted_user_ids == 162
    ), f"Found {num_posts_from_submitted_user_ids} SuicideWatch posts made by the 125 validation users of your json file, but was expecting 162."

    # Filtered data should have 162 SuicideWatch posts made by 125 users.
    # Check mapping of user_ids/post_ids
    check_mapping_of_ids(submission_data=submission_data)
    print(
        "Done: user_ids and post_ids provided in the submission (json) file match the expected ones."
    )
    print("Next: checking the structure of your submission (json) file.")

    # Check fields in submitted data
    check_json_structure(submission_data=submission_data, data=merged_df)

    print("Done!")


import sys
if __name__ == "__main__":

    # Define the path for the data and json file
    csv_expert_path = "umd_reddit_suicidewatch_dataset_v2/expert/expert.csv"
    csv_expert_posts_path = "umd_reddit_suicidewatch_dataset_v2/expert/expert_posts.csv"
    submission_json_path = sys.argv[1]

    main(csv_expert_path, csv_expert_posts_path, submission_json_path)
