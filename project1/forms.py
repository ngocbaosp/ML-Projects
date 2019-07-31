from wtforms import StringField, SubmitField, SelectMultipleField, SelectField, IntegerField
from flask_wtf import FlaskForm
from wtforms.validators import DataRequired, NumberRange


class SearchForm(FlaskForm):
    genre = SelectMultipleField('Genre', validators=[DataRequired],
                                choices=[('Adventure', 'Adventure'), ('Drama', 'Drama'), ('Fantasy', 'Fantasy'),
                                         ('Horror', 'Horror'), ('Sci-Fi', 'Sci-Fi'), ('Supernatural', 'Supernatural'),
                                         ('Action', 'Action'), ('Kids', 'Kids'), ('Comedy', 'Comedy')])
    type = SelectField('Type', validators=[DataRequired],
                       choices=[('Movie', 'Movie'), ('Music', 'Music'), ('ONA', 'ONA'), ('OVA', 'OVA'),
                                ('Special', 'Special'), ('TV', 'TV')])
    episode = IntegerField('Number of episode', validators=[DataRequired, NumberRange(min=1, max=200)])
    members = IntegerField('Number of members', validators=[DataRequired])
    submit = SubmitField('Predict')
