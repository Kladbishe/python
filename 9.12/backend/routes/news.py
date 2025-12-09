from flask import Blueprint, render_template, request, redirect, url_for, jsonify
from extensions import db
from models import Article

news_bp = Blueprint('news', __name__, template_folder='../templates/news')

@news_bp.route('/')
def index():
    articles = Article.query.filter_by(category='news').order_by(Article.created_at.desc()).all()
    return render_template('news/index.html', articles=articles, category='News')

@news_bp.route('/<int:article_id>')
def view_article(article_id):
    article = Article.query.get_or_404(article_id)
    if article.category != 'news':
        return redirect(url_for('news.index'))
    return render_template('news/article.html', article=article)

@news_bp.route('/create', methods=['GET', 'POST'])
def create_article():
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        author = request.form.get('author')

        if title and content and author:
            article = Article(
                title=title,
                content=content,
                category='news',
                author=author
            )
            db.session.add(article)
            db.session.commit()
            return redirect(url_for('news.index'))

    return render_template('news/create.html')

@news_bp.route('/api')
def api_list():
    articles = Article.query.filter_by(category='news').order_by(Article.created_at.desc()).all()
    return jsonify([article.to_dict() for article in articles])
