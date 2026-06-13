import pandas as pd, numpy as np, warnings, os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
    r2_score, f1_score, roc_auc_score, precision_recall_fscore_support)
from imblearn.over_sampling import SMOTE
from xgboost import XGBRegressor
warnings.filterwarnings('ignore')

class DataProcessor:
    RFM_FEATURES = ['Recency', 'Frequency', 'MntTotal']
    CLV_FEATURES = ['Income','Age','Recency','Frequency','MntTotal',
        'R_Score','F_Score','M_Score','RFM_Score','AvgBasketSize',
        'DigitalEngagement','DealSensitivity','CampaignEngagement','HasChildren','CustomerSeniority']
    CLF_FEATURES = CLV_FEATURES + ['CLV_Proxy']

    def __init__(self):
        self.df_clean = None; self.df_sprint2 = None; self.df_sprint3 = None
        self.scaler_clv = StandardScaler(); self.scaler_clf = StandardScaler()
        self.scaler_rfm = StandardScaler(); self.models = {}

    def load_data(self, path):
        self.df_clean = pd.read_csv(path)
        self.df_clean.columns = self.df_clean.columns.str.strip()
        return self.df_clean

    def run_clustering(self, n=2):
        df = self.df_clean.copy()
        X = self.scaler_rfm.fit_transform(df[self.RFM_FEATURES].values)
        km = KMeans(n_clusters=n, init='k-means++', n_init=20, max_iter=500, random_state=42)
        df['Cluster'] = km.fit_predict(X)
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        sil = silhouette_score(X, km.labels_); db = davies_bouldin_score(X, km.labels_)
        hc = df.groupby('Cluster')['MntTotal'].mean().idxmax()
        df['Segment'] = df['Cluster'].apply(lambda x: 'Clients à Haute Valeur' if x == hc else 'Clients Sensibles aux Promotions')
        self.models['kmeans'] = km; self.models['kmeans_metrics'] = {'silhouette': sil, 'davies_bouldin': db}
        self.df_sprint2 = df; return df, sil, db

    def train_clv_model(self):
        df = self.df_sprint2.copy(); X = df[self.CLV_FEATURES].values; y = df['CLV_Proxy'].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        xgb = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0)
        xgb.fit(Xtr, ytr); yp = xgb.predict(Xte)
        m = {'MAE': mean_absolute_error(yte,yp), 'RMSE': np.sqrt(mean_squared_error(yte,yp)), 'R2': r2_score(yte,yp)}
        cv = cross_val_score(xgb, X, y, cv=5, scoring='r2'); m['R2_CV_mean'] = cv.mean(); m['R2_CV_std'] = cv.std()
        df['CLV_pred_XGBoost'] = xgb.predict(X)
        self.scaler_clv.fit(Xtr); lr = LinearRegression(); lr.fit(self.scaler_clv.transform(Xtr), ytr); ylr = lr.predict(self.scaler_clv.transform(Xte))
        ml = {'MAE': mean_absolute_error(yte,ylr), 'RMSE': np.sqrt(mean_squared_error(yte,ylr)), 'R2': r2_score(yte,ylr)}
        cvl = cross_val_score(lr, self.scaler_clv.transform(X), y, cv=5, scoring='r2'); ml['R2_CV_mean'] = cvl.mean(); ml['R2_CV_std'] = cvl.std()
        self.models['xgb_clv'] = xgb; self.models['lr_clv'] = lr; self.models['xgb_clv_metrics'] = m; self.models['lr_clv_metrics'] = ml
        self.df_sprint2 = df; return df, m, ml

    def train_classification_model(self):
        df = self.df_sprint2.copy(); X = df[self.CLF_FEATURES].values; y = df['Response'].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        sm = SMOTE(random_state=42, k_neighbors=5); Xs, ys = sm.fit_resample(Xtr, ytr)
        def _m(yt, yp, ypr):
            return {'F1': f1_score(yt,yp), 'ROC_AUC': roc_auc_score(yt,ypr),
                'Precision': precision_recall_fscore_support(yt,yp)[0][1],
                'Recall': precision_recall_fscore_support(yt,yp)[1][1], 'Accuracy': (yp==yt).mean()}
        rf_s = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)
        rf_s.fit(Xs, ys); ms = _m(yte, rf_s.predict(Xte), rf_s.predict_proba(Xte)[:,1])
        rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, class_weight='balanced', random_state=42, n_jobs=-1)
        rf.fit(Xtr, ytr); mf = _m(yte, rf.predict(Xte), rf.predict_proba(Xte)[:,1])
        self.scaler_clf.fit(Xtr); Xss = self.scaler_clf.transform(Xs); Xts = self.scaler_clf.transform(Xte)
        lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0); lr.fit(Xss, ys); ml = _m(yte, lr.predict(Xts), lr.predict_proba(Xts)[:,1])
        df['P_Response'] = rf.predict_proba(X)[:,1]
        self.models['rf_smote'] = rf_s; self.models['rf'] = rf; self.models['lr_clf'] = lr
        self.models['rf_smote_metrics'] = ms; self.models['rf_metrics'] = mf; self.models['lr_clf_metrics'] = ml
        self.df_sprint2 = df; return df, ms, mf, ml

    def calculate_unified_score(self):
        df = self.df_sprint2.copy(); df['Score_Unifie'] = df['P_Response'] * df['CLV_pred_XGBoost']
        df['Score_Unifie_Norm'] = MinMaxScaler().fit_transform(df[['Score_Unifie']])
        df['Score_Decile'] = pd.qcut(df['Score_Unifie'], q=10, labels=False, duplicates='drop') + 1
        t = df['Score_Unifie'].quantile(0.80); df['Top20_Score'] = (df['Score_Unifie'] >= t).astype(int)
        self.df_sprint3 = df; return df

    def analyze_strategies(self, pct=0.20):
        df = self.df_sprint3.copy(); n = int(len(df)*pct); br = df['Response'].mean()
        strats = {'Ciblage Aléatoire': df.sample(n, random_state=42).index,
            'Ciblage CLV Seule': df.nlargest(n,'CLV_pred_XGBoost').index,
            'Ciblage P(Réponse) Seule': df.nlargest(n,'P_Response').index,
            'Ciblage Score Unifié': df.nlargest(n,'Score_Unifie').index}
        res = []; rc = None; rr = None
        for nm, idx in strats.items():
            s = df.loc[idx]; nr = s['Response'].sum(); ct = s['CLV_pred_XGBoost'].sum()
            p = nr/n; l = p/br; pr = nr*11-n*3; roi = pr/(n*3)*100
            if rc is None: rc = ct; rr = nr
            gc = ((ct-rc)/rc*100) if rc>0 else 0; gr = ((nr-rr)/rr*100) if rr>0 else 0
            res.append({'Stratégie':nm,'Répondants':int(nr),'Précision':round(p*100,1),'Lift':round(l,2),
                'CLV Total (€)':round(ct,2),'Profit Net (€)':round(pr,2),'ROI (%)':round(roi,1),'Gain CLV (%)':round(gc,1),'Gain Répondants (%)':round(gr,1)})
        self.models['strategy_results'] = pd.DataFrame(res); return pd.DataFrame(res)

    def run_full_pipeline(self, path='data/ifood_cleaned.csv'):
        self.load_data(path); self.run_clustering(); self.train_clv_model()
        self.train_classification_model(); self.calculate_unified_score(); self.analyze_strategies()
        return self.df_sprint3, self.models

    def get_segment_profile(self, name):
        df = self.df_sprint3; s = df[df['Segment']==name]
        if s.empty: return None
        cats = ['MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']
        dep = {c: round(s[c].mean(),2) for c in cats}
        return {'nom_segment':name,'taille':len(s),'pourcentage':round(len(s)/len(df)*100,1),
            'clv_moyen':round(s['CLV_pred_XGBoost'].mean(),2),'frequence_moyenne':round(s['Frequency'].mean(),1),
            'recence_moyenne':round(s['Recency'].mean(),1),'montant_total_moyen':round(s['MntTotal'].mean(),2),
            'panier_moyen':round(s['AvgBasketSize'].mean(),2),'score_reponse_moyen':round(s['P_Response'].mean(),3),
            'score_unifie_moyen':round(s['Score_Unifie'].mean(),2),'rfm_moyen':round(s['RFM_Score'].mean(),1),
            'revenu_moyen':round(s['Income'].mean(),2),'age_moyen':round(s['Age'].mean(),1),
            'engagement_digital':round(s['DigitalEngagement'].mean(),1),'engagement_campagne':round(s['CampaignEngagement'].mean(),2),
            'sensibilite_promo':round(s['DealSensitivity'].mean(),2),'taux_enfants':round(s['HasChildren'].mean()*100,1),
            'taux_reponse':round(s['Response'].mean()*100,1),'top20_count':int(s['Top20_Score'].sum()),
            'depenses_par_categorie':dep,'top_categorie':max(dep,key=dep.get)}

    def get_all_segments(self):
        return self.df_sprint3['Segment'].unique().tolist() if self.df_sprint3 is not None else []
