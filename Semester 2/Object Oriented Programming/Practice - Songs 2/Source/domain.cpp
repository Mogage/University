#include "domain.h"

#include <ostream>

Song::Song()
{
    Id = -1;
    Title = "";
    Artist = "";
    Gen = "";
}

Song::Song(int _Id, string _Title, string _Artist, string _Gen) :
    Id{ _Id }, Title{ _Title }, Artist{ _Artist }, Gen{ _Gen } {}

int Song::id() const
{
    return Id;
}

string Song::title() const 
{
    return Title;
}

string Song::artist() const
{
    return Artist;
}

string Song::gen() const
{
    return Gen;
}

std::ostream& operator<<(std::ostream& Os, const Song& _Song)
{
    Os << _Song.Id << ',' << _Song.Title << ',' << _Song.Artist << ',' << _Song.Gen << '\n';
    return Os;
}
