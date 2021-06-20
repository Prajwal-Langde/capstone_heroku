from flask import Flask, render_template, request
import pandas as pd
import model

#Loading Flask
app = Flask(__name__)

#Load all the necessary files
df = pd.read_csv("sample30.csv")
fin_ratings = pd.read_csv("./User_final_ratings.csv")
fin_ratings = fin_ratings.set_index('user_id')

@app.route("/", methods=['POST', "GET"])
def home():
    if request.method == 'POST':
        user = request.form['nm']
        if user in fin_ratings.index.tolist():
            
            #Select the top 20 recommendations
            lis = fin_ratings.loc[user].sort_values(ascending=False).index[:20]
            df_recom = df[df['name'].isin(lis)]
            fin_df = model.clean_reviews(df_recom)
            
            #Pick the top 5 items from that list based on positive sentiment percentage
            fin_df = fin_df[['name', 'preds']]
            d = fin_df.groupby('name').mean().sort_values(ascending=False, by="preds")*100
            products = d[:5].index.tolist()
            
            #Display those on the HTML page
            return render_template('index.html', products=products, submit="yes")
        else:
            return render_template('index.html', products="None")
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)