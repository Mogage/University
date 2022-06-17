#include "domain.h"

Song::Song()
{
	Id = Rank = -1;
	Title = Artist = "";
}

Song::Song(int _Id, string _Title, string _Artist, int _Rank) :
	Id{ _Id }, Title{ _Title }, Artist{ _Artist }, Rank{ _Rank }
{}

int Song::id() const
{
	return Id;
}

int Song::rank() const
{
	return Rank;
}

string Song::title() const
{
	return Title;
}

string Song::artist() const
{
	return Artist;
}

void Song::setRank(int _Rank)
{
	Rank = _Rank;
}

void Song::setTitle(string _Title)
{
	Title = _Title;
}

std::ostream& operator<<(std::ostream& Os, const Song& ToPrint)
{
	Os << ToPrint.id() << ',' << ToPrint.title() << ',' << ToPrint.artist() << ',' << ToPrint.rank() << '\n';
	return Os;
}
