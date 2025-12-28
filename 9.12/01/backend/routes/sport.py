from flask import Blueprint, render_template, request, redirect, url_for, jsonify
from extensions import db
from models import Article

sport_bp = Blueprint('sport', __name__, template_folder='../templates/sport')

@sport_bp.route('/')
def index():
    articles = Article.query.filter_by(category='sport').order_by(Article.created_at.desc()).all()
    return render_template('sport/index.html', articles=articles, category='Sport')

@sport_bp.route('/<int:article_id>')
def view_article(article_id):
    article = Article.query.get_or_404(article_id)
    if article.category != 'sport':
        return redirect(url_for('sport.index'))
    return render_template('sport/article.html', article=article)

@sport_bp.route('/create', methods=['GET', 'POST'])
def create_article():
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        author = request.form.get('author')

        if title and content and author:
            article = Article(
                title=title,
                content=content,
                category='sport',
                author=author
            )
            db.session.add(article)
            db.session.commit()
            return redirect(url_for('sport.index'))

    return render_template('sport/create.html')

@sport_bp.route('/api')
def api_list():
    articles = Article.query.filter_by(category='sport').order_by(Article.created_at.desc()).all()
    return jsonify([article.to_dict() for article in articles])
