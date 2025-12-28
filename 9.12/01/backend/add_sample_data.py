from app import create_app
from extensions import db
from models import Article
from datetime import datetime, timedelta

app = create_app()

with app.app_context():
    existing = Article.query.first()
    if existing:
        print("Articles already exist in database")

    news_articles = [
        {
            'title': 'Breaking: New Technology Revolutionizes Renewable Energy',
            'content': '''Scientists at MIT have announced a groundbreaking discovery in solar panel technology that could increase efficiency by 40%. The new photovoltaic cells use a novel material composition that captures a broader spectrum of light.

This breakthrough could significantly reduce the cost of solar energy and accelerate the transition to renewable power sources worldwide. The research team expects commercial production to begin within two years.

Industry experts are calling this one of the most significant advances in clean energy technology in the past decade.''',
            'category': 'news',
            'author': 'Sarah Johnson'
        },
        {
            'title': 'Global Climate Summit Reaches Historic Agreement',
            'content': '''World leaders have signed a comprehensive climate accord at the International Climate Summit, committing to reduce carbon emissions by 50% by 2030.

The agreement includes provisions for:
- Increased funding for renewable energy projects
- Stricter regulations on industrial emissions
- Protection of endangered ecosystems
- Technology transfer to developing nations

Environmental groups have praised the agreement as a crucial step in addressing climate change, though some critics argue the targets should be even more ambitious.''',
            'category': 'news',
            'author': 'Michael Chen'
        },
        {
            'title': 'City Announces Major Infrastructure Upgrade Plan',
            'content': '''The city council has approved a $2 billion infrastructure modernization project that will upgrade roads, bridges, and public transportation over the next five years.

Key components of the plan include:
- Repair of 150 miles of roads
- Modernization of the metro system
- Construction of new bike lanes
- Smart traffic management systems

Mayor officials expect the project to create thousands of jobs while improving quality of life for residents.''',
            'category': 'news',
            'author': 'Emily Rodriguez'
        }
    ]

    sport_articles = [
        {
            'title': 'Championship Final: Underdogs Claim Victory in Stunning Upset',
            'content': '''In a dramatic turn of events, the underdog team secured a 3-2 victory in the championship final, ending their opponent\'s undefeated season.

The match was decided in the final minutes when striker James Wilson scored a spectacular goal, sending fans into celebration. This marks the team\'s first championship title in 25 years.

Coach Martinez praised his team\'s determination: "They never gave up, even when we were down 2-1. This victory is testament to their hard work and dedication throughout the season."''',
            'category': 'sport',
            'author': 'David Thompson'
        },
        {
            'title': 'Olympic Athlete Breaks World Record',
            'content': '''Track star Maria Santos shattered the 100-meter world record at the National Championships, clocking an incredible 10.42 seconds.

The previous record had stood for eight years. Santos, 24, becomes the youngest athlete to hold this record in the event\'s history.

"I can\'t believe it," Santos said after the race. "All the training and sacrifices have paid off. I\'m looking forward to representing my country at the Olympics."

Her coach attributes the success to a revolutionary training program combining traditional methods with cutting-edge sports science.''',
            'category': 'sport',
            'author': 'Lisa Anderson'
        }
    ]

    economic_articles = [
        {
            'title': 'Stock Market Reaches Record High Amid Tech Boom',
            'content': '''Major stock indices reached all-time highs today, driven by strong earnings reports from technology companies and optimistic economic forecasts.

The tech sector led the rally, with several major companies reporting quarterly earnings that exceeded analyst expectations. Artificial intelligence and cloud computing companies showed particularly strong growth.

Market analysts attribute the gains to:
- Robust consumer spending
- Declining inflation rates
- Positive employment data
- Increased business investment

However, some experts caution that valuations in certain sectors may be reaching unsustainable levels.''',
            'category': 'economic',
            'author': 'Robert Martinez'
        },
        {
            'title': 'Central Bank Announces Interest Rate Decision',
            'content': '''The Federal Reserve announced it will maintain current interest rates, citing stable economic growth and controlled inflation.

In a statement, the Fed noted that the economy continues to expand at a moderate pace, with unemployment remaining low and wage growth steady.

Key points from the announcement:
- Interest rates held at 4.5-4.75%
- GDP growth projected at 2.1% for the year
- Inflation moving toward 2% target
- Labor market remains strong

Financial markets responded positively to the news, with major indices rising in afternoon trading. Economists expect rates to remain stable for the remainder of the year.''',
            'category': 'economic',
            'author': 'Jennifer Lee'
        },
        {
            'title': 'Startup Sector Sees Record Investment in Q1',
            'content': '''Venture capital investment in startups reached $85 billion in the first quarter, marking a 35% increase compared to last year.

Technology startups, particularly in AI and biotechnology, attracted the largest share of funding. Several "unicorn" companies emerged, achieving valuations over $1 billion.

Industry trends:
- AI/ML companies: $28 billion
- Healthcare tech: $19 billion
- Fintech: $15 billion
- Clean energy: $12 billion

Investors remain optimistic about innovation-driven sectors despite broader economic uncertainties. Many are betting on technologies that promise to transform traditional industries.''',
            'category': 'economic',
            'author': 'Kevin Park'
        }
    ]

    all_articles = news_articles + sport_articles + economic_articles

    for i, article_data in enumerate(all_articles):
        article = Article(
            title=article_data['title'],
            content=article_data['content'],
            category=article_data['category'],
            author=article_data['author'],
            created_at=datetime.utcnow() - timedelta(hours=len(all_articles)-i)
        )
        db.session.add(article)

    db.session.commit()
    print(f"Successfully added {len(all_articles)} articles to the database!")
    print(f"- {len(news_articles)} news articles")
    print(f"- {len(sport_articles)} sport articles")
    print(f"- {len(economic_articles)} economic articles")
