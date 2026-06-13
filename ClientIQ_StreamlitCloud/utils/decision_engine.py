"""
ClientIQ v3 — DecisionEngine
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from utils.marketing_agent import MarketingAgent

class DecisionEngine:
    CAT_LABELS = {
        'MntWines': 'Vins & Boissons', 'MntFruits': 'Fruits & Légumes',
        'MntMeatProducts': 'Produits Carnés', 'MntFishProducts': 'Produits de la Mer',
        'MntSweetProducts': 'Produits Sucrés', 'MntGoldProds': 'Produits Premium / Or',
    }
    PROFILES = {
        'ambassadeur_or': {
            'titre': 'Ambassadeur Or', 'icone': '👑', 'couleur': '#F59E0B',
            'description': 'Client premium avec forte valeur, haute fidélité et engagement exceptionnel. Priorité maximale.',
            'actions': [
                'Accorder le statut VIP avec avantages exclusifs',
                'Proposer un conseiller personnel dédié',
                'Inviter aux événements privés de la marque',
                'Offrir un accès anticipé aux nouvelles collections',
                'Programme de fidélité sur-mesure avec récompenses premium',
            ],
            'canal': 'Email personnalisé + Conseiller dédié', 'frequence': '2 fois/semaine',
            'budget_priorite': 'Élevé', 'roi_attendu': 'Très élevé (>300%)',
        },
        'client_fidele': {
            'titre': 'Client Fidèle', 'icone': '⭐', 'couleur': '#2DD4BF',
            'description': 'Client régulier avec bon historique. Candidat idéal pour cross-selling et upselling.',
            'actions': [
                'Programme de parrainage avec récompenses mutuelles',
                'Offres groupées basées sur l\'historique d\'achat',
                'Recommandations personnalisées par catégorie préférée',
                'Suivi post-achat avec enquête de satisfaction',
            ],
            'canal': 'Email recommandation + Notification push', 'frequence': '1 fois/semaine',
            'budget_priorite': 'Moyen-Élevé', 'roi_attendu': 'Élevé (200-300%)',
        },
        'chasseur_promos': {
            'titre': 'Chasseur de Promos', 'icone': '🏷️', 'couleur': '#FB923C',
            'description': 'Motivé par les réductions. Réagit bien aux offres limitées, nécessite des incitations régulières.',
            'actions': [
                'Programmer des ventes flash régulières (48h max)',
                'Offres type "achetez 2, le 3ème offert"',
                'Alertes SMS pour les soldes et promotions',
                'Programme de cashback progressif',
            ],
            'canal': 'SMS promotionnel + Réseaux sociaux', 'frequence': '3 fois/semaine',
            'budget_priorite': 'Faible', 'roi_attendu': 'Modéré (100-150%)',
        },
        'a_reactiver': {
            'titre': 'Client à Réactiver', 'icone': '⏰', 'couleur': '#F43F5E',
            'description': 'Client inactif ou en perte de vitesse. Campagne de réactivation urgente nécessaire.',
            'actions': [
                'Campagne de réactivation avec -30% de réduction',
                'Offrir la livraison gratuite sur la prochaine commande',
                'Email "Vous nous manquez" avec incentive forte',
                'Code promo à durée limitée (72h)',
            ],
            'canal': 'Email avec incentive + Retargeting', 'frequence': 'Immédiat puis J+3, J+7',
            'budget_priorite': 'Élevé (urgence)', 'roi_attendu': 'Variable (50-200%)',
        },
        'en_croissance': {
            'titre': 'Pépite en Croissance', 'icone': '📈', 'couleur': '#38BDF8',
            'description': 'Bon potentiel sous-exploité. Focus sur augmentation panier et fréquence d\'achat.',
            'actions': [
                'Packs découverte à prix attractifs',
                'Programme de récompenses au panier progressif',
                'Recommandations basées sur goûts similaires',
                'Avantage après le 3ème achat consécutif',
            ],
            'canal': 'Email + Publicité ciblée', 'frequence': '2 fois/semaine',
            'budget_priorite': 'Moyen', 'roi_attendu': 'Bon (150-250%)',
        },
        'client_nouveau': {
            'titre': 'Nouveau Client', 'icone': '🚀', 'couleur': '#C084FC',
            'description': 'Client récent avec peu d\'historique. La première expérience est déterminante.',
            'actions': [
                'Email de bienvenue avec code -15%',
                'Livraison gratuite sur la 1ère commande',
                'Guide d\'achat personnalisé',
                'Onboarding structuré : J0, J3, J7, J14',
            ],
            'canal': 'Email bienvenue + SMS', 'frequence': 'Onboarding (J0-J14)',
            'budget_priorite': 'Investissement', 'roi_attendu': 'Long terme (200-400%)',
        },
    }

    def __init__(self, df, models, processor):
        self.df = df.copy()
        self.models = models
        self.processor = processor
        self.marketing_agent = MarketingAgent()
        self._precompute()

    def _precompute(self):
        self.stats = {
            'clv_median': self.df['CLV_pred_XGBoost'].median(),
            'clv_mean': self.df['CLV_pred_XGBoost'].mean(),
            'clv_std': self.df['CLV_pred_XGBoost'].std(),
            'clv_q75': self.df['CLV_pred_XGBoost'].quantile(0.75),
            'freq_mean': self.df['Frequency'].mean(),
            'rec_mean': self.df['Recency'].mean(),
            'panier_mean': self.df['AvgBasketSize'].mean(),
            'rfm_mean': self.df['RFM_Score'].mean(),
            'p_resp_mean': self.df['P_Response'].mean(),
            'score_uni_mean': self.df['Score_Unifie'].mean(),
            'income_mean': self.df['Income'].mean(),
            'age_mean': self.df['Age'].mean(),
        }

    def classify(self, c):
        clv = c.get('CLV_pred_XGBoost', 0); freq = c.get('Frequency', 0)
        rec = c.get('Recency', 999); pr = c.get('P_Response', 0)
        sens = c.get('DealSensitivity', 0); eng = c.get('DigitalEngagement', 0)
        sen = c.get('CustomerSeniority', 0); pan = c.get('AvgBasketSize', 0)
        rfm = c.get('RFM_Score', 0)
        if clv >= 100 and freq >= 15 and rec <= 20: return 'ambassadeur_or'
        if clv >= 50 and freq >= 8 and pr >= 0.2 and rec <= 60:
            return 'ambassadeur_or' if eng >= 12 else 'client_fidele'
        if sens >= 0.20 and clv < 50: return 'chasseur_promos'
        if rec > 60 and freq < 8: return 'a_reactiver'
        if sen <= 6 and freq <= 8: return 'client_nouveau'
        if clv >= 25 or (rfm >= 12 and pan >= 40): return 'en_croissance'
        return 'client_fidele' if clv >= 50 else 'chasseur_promos'

    def fidelity(self, c):
        sc = 0; r = c.get('Recency', 999); f = c.get('Frequency', 0)
        clv = c.get('CLV_pred_XGBoost', 0); pr = c.get('P_Response', 0)
        eng = c.get('CampaignEngagement', 0)
        if r <= 20: sc += 25
        elif r <= 40: sc += 18
        elif r <= 60: sc += 10
        else: sc += 3
        if f >= 15: sc += 25
        elif f >= 8: sc += 16
        elif f >= 4: sc += 8
        else: sc += 2
        if clv >= 100: sc += 25
        elif clv >= 50: sc += 16
        elif clv >= 25: sc += 8
        else: sc += 2
        if pr >= 0.5: sc += 15 + (10 if eng >= 0.5 else 5)
        elif pr >= 0.2: sc += 10 + (5 if eng >= 0.3 else 0)
        else: sc += 3
        if sc >= 80: return 'Fidèle Premium', sc, '#F59E0B'
        if sc >= 60: return 'Fidèle', sc, '#2DD4BF'
        if sc >= 40: return 'Neutre', sc, '#FBBF24'
        if sc >= 20: return 'A Risque', sc, '#FB923C'
        return 'En Danger', sc, '#F43F5E'

    def percentiles(self, c):
        metrics = {'CLV': 'CLV_pred_XGBoost', 'Fréquence': 'Frequency', 'Récence': 'Recency',
                   'Panier': 'AvgBasketSize', 'RFM': 'RFM_Score', 'P(Réponse)': 'P_Response',
                   'Score Uni': 'Score_Unifie', 'Revenu': 'Income'}
        out = {}
        for label, col in metrics.items():
            val = c.get(col, 0); pct = (self.df[col].values < val).sum() / len(self.df) * 100
            mean = self.df[col].mean(); diff = ((val - mean) / max(abs(mean), 0.01)) * 100
            out[label] = {'val': round(val, 2), 'pct': round(pct, 1), 'mean': round(mean, 2), 'diff': round(diff, 1)}
        return out

    def spending(self, c):
        cats = list(self.CAT_LABELS.keys()); total = sum(c.get(cat, 0) for cat in cats)
        out = {}
        for cat in cats:
            v = c.get(cat, 0); p = (v / total * 100) if total > 0 else 0
            avg = self.df[cat].mean()
            out[self.CAT_LABELS[cat]] = {'montant': round(v, 2), 'pct': round(p, 1), 'avg': round(avg, 2), 'ratio': round(v / max(avg, 0.01), 2)}
        out['total'] = round(total, 2)
        out['top'] = max((k for k in out if k not in ['total', 'top']), key=lambda k: out[k]['montant']) if total > 0 else 'N/A'
        return out

    def full_profile(self, idx):
        c = self.df.iloc[idx].to_dict(); pk = self.classify(c)
        pi = self.PROFILES[pk]; fn, fs, fc = self.fidelity(c)
        return {'idx': idx, 'client': c, 'pk': pk, 'pi': pi,
                'fidelity': {'name': fn, 'score': fs, 'color': fc},
                'pct': self.percentiles(c), 'spend': self.spending(c)}

    def score_new_client(self, data_dict):
        c = {k: float(v) for k, v in data_dict.items() if v != '' and v is not None}
        cats = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        c['MntTotal'] = sum(c.get(cat, 0) for cat in cats)
        c['Frequency'] = c.get('NumWebPurchases', 0) + c.get('NumCatalogPurchases', 0) + c.get('NumStorePurchases', 0)
        c['AvgBasketSize'] = c['MntTotal'] / max(c['Frequency'], 1)
        rec = c.get('Recency', 50); freq = c.get('Frequency', 0); mnt = c.get('MntTotal', 0)
        c['R_Score'] = max(1, min(5, 5 - rec / 12.5)) if rec < 63 else 1
        c['F_Score'] = 5 if freq > 20 else 4 if freq > 10 else 3 if freq > 5 else 2 if freq > 0 else 1
        c['M_Score'] = 5 if mnt > 1000 else 4 if mnt > 500 else 3 if mnt > 200 else 2 if mnt > 50 else 1
        c['RFM_Score'] = c['R_Score'] + c['F_Score'] + c['M_Score']
        web_p = c.get('NumWebPurchases', 0); web_v = c.get('NumWebVisitsMonth', 0)
        camp = c.get('AcceptedCmpTotal', 0); deals = c.get('NumDealsPurchases', 0)
        c['DigitalEngagement'] = web_p + web_v
        c['DealSensitivity'] = deals / max(c['Frequency'], 1)
        c['CampaignEngagement'] = camp / 6
        c['HasChildren'] = float(c.get('Kidhome', 0) > 0 or c.get('Teenhome', 0) > 0)
        c['CustomerSeniority'] = c.get('DaysAsCustomer', 0) / 30
        c['CLV_Proxy'] = c['MntTotal'] / max(c['CustomerSeniority'], 1)
        from utils.data_processor import DataProcessor
        feats_clv = [c.get(f, 0) for f in DataProcessor.CLV_FEATURES]
        xgb = self.models.get('xgb_clv')
        if xgb:
            try: c['CLV_pred_XGBoost'] = float(xgb.predict(self.processor.scaler_clv.transform([feats_clv]))[0])
            except: c['CLV_pred_XGBoost'] = c['CLV_Proxy']
        else: c['CLV_pred_XGBoost'] = c['CLV_Proxy']
        rf = self.models.get('rf')
        if rf:
            from utils.data_processor import DataProcessor
            feats_clf = [c.get(f, 0) for f in DataProcessor.CLF_FEATURES]
            try: c['P_Response'] = float(rf.predict_proba(self.processor.scaler_clf.transform([feats_clf]))[0][1])
            except: c['P_Response'] = self.stats['p_resp_mean']
        else: c['P_Response'] = self.stats['p_resp_mean']
        c['Score_Unifie'] = c['CLV_pred_XGBoost'] * c['P_Response']
        c['Score_Unifie_Norm'] = 0
        c['Top20_Score'] = 1 if c['Score_Unifie'] >= self.df['Score_Unifie'].quantile(0.80) else 0
        pk = self.classify(c); pi = self.PROFILES[pk]
        fn, fs, fc = self.fidelity(c)
        km = self.models.get('kmeans')
        if km:
            from utils.data_processor import DataProcessor
            rfm_vals = [c.get(f, 0) for f in DataProcessor.RFM_FEATURES]
            cluster = int(km.predict(self.processor.scaler_rfm.transform([rfm_vals]))[0])
            hc = self.df.groupby('Cluster')['MntTotal'].mean().idxmax()
            c['Segment'] = 'Clients à Haute Valeur' if cluster == hc else 'Clients Sensibles aux Promotions'
        else:
            c['Segment'] = 'Clients à Haute Valeur' if c['CLV_pred_XGBoost'] >= self.stats['clv_median'] else 'Clients Sensibles aux Promotions'
        c['Cluster'] = c.get('Cluster', 0)
        return {'client': c, 'profil_key': pk, 'profil': pi,
                'fidelity': {'name': fn, 'score': fs, 'color': fc},
                'is_top20': bool(c.get('Top20_Score', 0)),
                'actions': pi['actions'], 'canal': pi['canal'],
                'budget_priorite': pi['budget_priorite'], 'roi_attendu': pi['roi_attendu']}

    def score_new_clients_batch(self, uploaded_df):
        results = []
        for _, row in uploaded_df.iterrows():
            try: results.append({'original': row.to_dict(), 'scored': self.score_new_client(row.to_dict())})
            except Exception as e: results.append({'original': row.to_dict(), 'error': str(e)})
        return results

    def compare(self, i1, i2):
        c1, c2 = self.df.iloc[i1].to_dict(), self.df.iloc[i2].to_dict()
        pk1, pk2 = self.classify(c1), self.classify(c2)
        f1, f2 = self.fidelity(c1), self.fidelity(c2)
        metrics = [('Revenu','Income','€'),('Âge','Age','ans'),('CLV','CLV_pred_XGBoost','€'),
                   ('P(Rép)','P_Response',''),('Score Uni','Score_Unifie',''),('RFM','RFM_Score','/15'),
                   ('Fréquence','Frequency',''),('Récence','Recency','j'),('Panier','AvgBasketSize','€'),
                   ('Digital','DigitalEngagement',''),('Montant Total','MntTotal','€')]
        comp = {}
        for label, key, unit in metrics:
            v1, v2 = c1.get(key, 0), c2.get(key, 0)
            d = v1 - v2; p = (d / max(abs(v2), 0.01)) * 100
            comp[label] = {'c1': round(v1, 2), 'c2': round(v2, 2), 'diff': round(d, 2), 'pct': round(p, 1),
                           'unit': unit, 'w': 1 if d > 0 else 2 if d < 0 else 0}
        return {'c1': c1, 'c2': c2, 'i1': i1, 'i2': i2, 'comp': comp,
                'pk1': pk1, 'pk2': pk2, 'pi1': self.PROFILES[pk1], 'pi2': self.PROFILES[pk2],
                'f1': f1, 'f2': f2, 'sp1': self.spending(c1), 'sp2': self.spending(c2)}

    def simulate(self, idx, mods):
        from copy import deepcopy
        c = deepcopy(self.df.iloc[idx].to_dict())
        for k, v in mods.items():
            if v != '' and v is not None:
                try: c[k] = float(v)
                except: continue
        cats = list(self.CAT_LABELS.keys())
        c['MntTotal'] = sum(c.get(cat, 0) for cat in cats)
        c['Frequency'] = c.get('NumWebPurchases', 0) + c.get('NumCatalogPurchases', 0) + c.get('NumStorePurchases', 0)
        c['AvgBasketSize'] = c['MntTotal'] / max(c['Frequency'], 1)
        r = 5 - c.get('Recency', 50) / 12.5 if c.get('Recency', 50) < 63 else 1
        f = 5 if c.get('Frequency', 0) > 20 else 4 if c.get('Frequency', 0) > 10 else 3 if c.get('Frequency', 0) > 5 else 2
        m = 5 if c.get('MntTotal', 0) > 1000 else 4 if c.get('MntTotal', 0) > 500 else 3 if c.get('MntTotal', 0) > 200 else 2
        c['RFM_Score'] = r + f + m
        from utils.data_processor import DataProcessor
        xgb = self.models.get('xgb_clv')
        if xgb:
            feats = [c.get(f, 0) for f in DataProcessor.CLV_FEATURES]
            c['CLV_pred_XGBoost'] = float(xgb.predict(self.processor.scaler_clv.transform([feats]))[0])
        rf = self.models.get('rf')
        if rf:
            feats = [c.get(f, 0) for f in DataProcessor.CLF_FEATURES]
            c['P_Response'] = float(rf.predict_proba(self.processor.scaler_clf.transform([feats]))[0][1])
        c['Score_Unifie'] = c.get('CLV_pred_XGBoost', 0) * c.get('P_Response', 0)
        orig = self.df.iloc[idx].to_dict()
        delta = {}
        for key in ['CLV_pred_XGBoost', 'P_Response', 'Score_Unifie', 'RFM_Score', 'Frequency', 'AvgBasketSize']:
            ov, nv = orig.get(key, 0), c.get(key, 0)
            delta[key] = {'avant': round(ov, 3), 'apres': round(nv, 3), 'delta': round(nv - ov, 3),
                          'pct': round((nv - ov) / max(abs(ov), 0.01) * 100, 1)}
        old_pk, new_pk = self.classify(orig), self.classify(c)
        _, old_f, _ = self.fidelity(orig)
        new_fn, new_fs, new_fc = self.fidelity(c)
        return {'mods': mods, 'client': c, 'delta': delta,
                'old_pk': old_pk, 'new_pk': new_pk, 'old_f': old_f, 'new_f': (new_fn, new_fs),
                'changed': old_pk != new_pk or old_f != new_fn,
                'old_pi': self.PROFILES[old_pk], 'new_pi': self.PROFILES[new_pk]}

    def generate_campaign(self, segment_name=None, profile_key=None, budget_total=1000):
        if segment_name:
            clients = self.df[self.df['Segment'] == segment_name].copy()
        elif profile_key:
            clients = pd.DataFrame()
            for idx in range(len(self.df)):
                if self.classify(self.df.iloc[idx].to_dict()) == profile_key:
                    clients = pd.concat([clients, self.df.iloc[[idx]]])
        else:
            clients = self.df.nlargest(int(len(self.df) * 0.2), 'Score_Unifie')
        if len(clients) == 0: return None
        clients_sorted = clients.nlargest(len(clients), 'Score_Unifie')
        pi = self.PROFILES.get(self.classify(clients_sorted.iloc[0].to_dict()), self.PROFILES['client_fidele'])
        n_target = min(len(clients_sorted), max(10, int(budget_total / 3)))
        targeted = clients_sorted.head(n_target)
        est_resp = targeted['P_Response'].mean(); est_clv = targeted['CLV_pred_XGBoost'].mean()
        est_conv = est_resp * 0.9; est_rev = n_target * est_conv * est_clv
        est_profit = est_rev - budget_total
        est_roi = (est_profit / budget_total * 100) if budget_total > 0 else 0
        priority = 'HAUTE' if est_roi > 100 else 'MOYENNE' if est_roi > 30 else 'FAIBLE'
        urgency = 'IMMÉDIATE' if any(self.fidelity(c.to_dict())[0] in ['A Risque', 'En Danger'] for _, c in targeted.iterrows()) else 'PLANIFIÉE'
        client_list = []
        for idx, row in targeted.iterrows():
            pk = self.classify(row.to_dict()); fn, fs, fc = self.fidelity(row.to_dict())
            client_list.append({'Index': int(idx), 'Revenu': round(row.get('Income', 0), 0),
                'CLV_Predit': round(row['CLV_pred_XGBoost'], 1), 'P_Reponse': round(row['P_Response'], 3),
                'Score_Unifie': round(row['Score_Unifie'], 1), 'Profil': self.PROFILES[pk]['titre'],
                'Fidelite': fn, 'Canal_Recommande': self.PROFILES[pk]['canal'],
                'Priorite': 'HAUTE' if row['Score_Unifie'] >= self.df['Score_Unifie'].quantile(0.9) else 'MOYENNE'})
        return {'nom': f"Campagne {pi['titre']} — {datetime.now().strftime('%d/%m/%Y')}",
            'segment': segment_name or 'Top 20% Score', 'profil_cle': pi['titre'],
            'nb_cibles': n_target, 'budget': budget_total, 'est_conversion': round(est_conv * 100, 1),
            'est_revenu': round(est_rev, 0), 'est_profit': round(est_profit, 0), 'est_roi': round(est_roi, 1),
            'priorite': priority, 'urgence': urgency, 'canal': pi['canal'], 'frequence': pi['frequence'],
            'actions': pi['actions'], 'clients': client_list, 'date': datetime.now().strftime('%d/%m/%Y %H:%M'),
            'profil_couleur': pi['couleur'], 'profil_description': pi['description']}

    def generate_marketing_message(self, campaign):
        """Genere un message marketing via LLM Claude pour une campagne."""
        msg, llm_used = self.marketing_agent.generer_message_campagne(
            profil_titre=campaign['profil_cle'],
            profil_description=campaign.get('profil_description', ''),
            profil_couleur=campaign.get('profil_couleur', '#0D7377'),
            nb_cibles=campaign['nb_cibles'],
            budget=campaign['budget'],
            est_conversion=campaign['est_conversion'],
            est_roi=campaign['est_roi'],
            canal=campaign['canal'],
            actions=campaign['actions']
        )
        return {'message': msg, 'llm_used': llm_used,
                'llm_model': self.marketing_agent.llm_model if llm_used else None}

    def get_decisions(self, top_n=50):
        decisions = []
        for idx in range(len(self.df)):
            c = self.df.iloc[idx].to_dict(); fn, fs, fc = self.fidelity(c)
            if fn in ['En Danger', 'A Risque']:
                pk = self.classify(c)
                decisions.append({'type': 'RÉACTIVATION URGENTE', 'priorite': 'CRITIQUE',
                    'client_idx': int(idx), 'profil': self.PROFILES[pk]['titre'], 'fidelite': fn,
                    'score_fidelite': fs, 'clv_perdu': round(c['CLV_pred_XGBoost'], 1),
                    'recence': c.get('Recency', 0), 'fidelity_color': fc,
                    'action': f"Client #{idx} — Inactif depuis {int(c.get('Recency',0))}j, CLV: {c['CLV_pred_XGBoost']:.0f}€. Réactivation immédiate.",
                    'canal': self.PROFILES[pk]['canal'], 'delai': '24-48h'})
        for idx in range(len(self.df)):
            c = self.df.iloc[idx].to_dict(); pk = self.classify(c)
            if pk == 'ambassadeur_or':
                fn2, fs2, fc2 = self.fidelity(c)
                decisions.append({'type': 'FIDÉLISATION PREMIUM', 'priorite': 'HAUTE',
                    'client_idx': int(idx), 'profil': 'Ambassadeur Or', 'fidelite': fn2,
                    'score_fidelite': fs2, 'clv_perdu': round(c['CLV_pred_XGBoost'], 1),
                    'recence': c.get('Recency', 0), 'fidelity_color': fc2,
                    'action': f"Client #{idx} — Ambassadeur (CLV: {c['CLV_pred_XGBoost']:.0f}€). Programme VIP.",
                    'canal': 'Email + Conseiller', 'delai': 'Semaine courante'})
        for idx in range(len(self.df)):
            c = self.df.iloc[idx].to_dict(); pk = self.classify(c)
            if pk == 'en_croissance':
                fn3, fs3, fc3 = self.fidelity(c)
                decisions.append({'type': 'ACCÉLÉRATION CROISSANCE', 'priorite': 'MOYENNE',
                    'client_idx': int(idx), 'profil': 'Pépite en Croissance', 'fidelite': fn3,
                    'score_fidelite': fs3, 'clv_perdu': round(c['CLV_pred_XGBoost'], 1),
                    'recence': c.get('Recency', 0), 'fidelity_color': fc3,
                    'action': f"Client #{idx} — Bon potentiel (CLV: {c['CLV_pred_XGBoost']:.0f}€). Cross-selling.",
                    'canal': 'Email + Pub ciblée', 'delai': '2 semaines'})
        prio_order = {'CRITIQUE': 0, 'HAUTE': 1, 'MOYENNE': 2, 'FAIBLE': 3}
        decisions.sort(key=lambda x: (prio_order.get(x['priorite'], 9), -x['clv_perdu']))
        return decisions[:top_n]

    def export_campaign_csv(self, campaign):
        return pd.DataFrame(campaign['clients'])

    def export_decisions_csv(self, decisions):
        return pd.DataFrame(decisions)
