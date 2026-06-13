"""
Moteur de Recommandation Marketing — ClientIQ (Feedback Pr. SAHID)
Integre un veritable appel LLM (Anthropic Claude) pour la generation
de messages marketing personnalises dans les campagnes ClientIQ.
"""
import os
import requests

class MarketingAgent:
    """Moteur de Recommandation Marketing — genere des messages personnalises via LLM Claude."""

    def __init__(self):
        self.llm_model = 'claude-sonnet-4-20250514'

    def _get_api_key(self):
        return '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'

    def _call_claude(self, prompt):
        api_key = self._get_api_key()
        if not api_key:
            return None
        try:
            response = requests.post(
                'https://api.anthropic.com/v1/messages',
                headers={
                    'x-api-key': api_key,
                    'content-type': 'application/json',
                    'anthropic-version': '2023-06-01'
                },
                json={
                    'model': self.llm_model,
                    'max_tokens': 500,
                    'messages': [{'role': 'user', 'content': prompt}]
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()['content'][0]['text']
        except Exception as e:
            print(f"[MarketingAgent] Erreur Claude: {e}")
            return None

    def generer_message_campagne(self, profil_titre, profil_description, profil_couleur,
                                  nb_cibles, budget, est_conversion, est_roi, canal, actions):
        api_key = self._get_api_key()
        if not api_key:
            return self._template_fallback(profil_titre), False

        prompt = f"""Vous êtes un expert en marketing digital pour une entreprise de vente au détail de produits alimentaires.

Campagne cible :
- Profil cible : {profil_titre} — {profil_description}
- Nombre de clients ciblés : {nb_cibles}
- Budget alloué : {budget} euros
- Conversion estimée : {est_conversion}%
- ROI estimé : {est_roi}%
- Canal recommandé : {canal}
- Actions recommandées : {', '.join(actions[:3])}

Générez un message marketing personnalisé de 4 à 6 lignes pour cette campagne. Le ton doit être professionnel, chaleureux et orienté vers l'action. Incluez un appel à l'action clair."""

        message = self._call_claude(prompt)
        if message:
            return message.strip(), True
        return self._template_fallback(profil_titre), False

    def _template_fallback(self, profil_titre):
        templates = {
            "Ambassadeur Or": "Cher client privilégié,\n\nNous vous remercions pour votre fidélité exceptionnelle. En tant que membre de notre programme VIP, profitez d'un accès exclusif à nos nouvelles collections et d'une remise de -15% avec le code VIP2026.\n\nCordialement,\nL'équipe Marketing",
            "Client Fidèle": "Bonjour !\n\nVotre fidélité mérite d'être récompensée. Découvrez nos offres personnalisées spécialement sélectionnées pour vous, avec un cadeau surprise dès 50 euros d'achat.\n\nÀ très vite,\nL'équipe Marketing",
            "Chasseur de Promos": "Alerte Promotion !\n\nVos marques préférées sont en promotion avec jusqu'à -40% de réduction. Stocks limités, valide 48h seulement !\n\nFaites vos achats maintenant,\nL'équipe Marketing",
            "Client à Réactiver": "Bon retour parmi nous !\n\nVous nous avez manqué ! Profitez de -25% sur votre prochaine commande + livraison offerte. Offre valable 72h avec le code RETOUR2026.\n\nÀ bientôt,\nL'équipe Marketing",
            "Pépite en Croissance": "Découvrez de nouvelles saveurs !\n\nNos packs découverte à -30% sont pensés pour vous. Combinez 3 produits et économisez davantage.\n\nBonne exploration,\nL'équipe Marketing",
            "Nouveau Client": "Bienvenue !\n\nNous sommes ravis de vous compter parmi nous. Profitez de -15% sur votre 1ère commande avec le code BIENVENUE2026 + livraison gratuite.\n\nL'équipe Marketing",
        }
        return templates.get(profil_titre, "Découvrez nos offres personnalisées sélectionnées pour vous ! Code promo : BIEN2026")
