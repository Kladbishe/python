from flask import Blueprint, render_template, request, redirect, url_for, jsonify
from extensions import db
from models import Article

economic_bp = Blueprint('economic', __name__, template_folder='../templates/economic')

@economic_bp.route('/')
def index():
    articles = Article.query.filter_by(category='economic').order_by(Article.created_at.desc()).all()
    return render_template('economic/index.html', articles=articles, category='Economic')

@economic_bp.route('/<int:article_id>')
def view_article(article_id):
    article = Article.query.get_or_404(article_id)
    if article.category != 'economic':
        return redirect(url_for('economic.index'))
    return render_template('economic/article.html', article=article)

@economic_bp.route('/create', methods=['GET', 'POST'])
def create_article():
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        author = request.form.get('author')

        if title and content and author:
            article = Article(
                title=title,
                content=content,
                category='economic',
                author=author
            )
            db.session.add(article)
            db.session.commit()
            return redirect(url_for('economic.index'))

    return render_template('economic/create.html')

@economic_bp.route('/api')
def api_list():
    articles = Article.query.filter_by(category='economic').order_by(Article.created_at.desc()).all()
    return jsonify([article.to_dict() for article in articles])
