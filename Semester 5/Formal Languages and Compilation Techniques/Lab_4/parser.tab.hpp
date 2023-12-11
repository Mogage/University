
/* A Bison parser, made by GNU Bison 2.4.1.  */

/* Skeleton interface for Bison's Yacc-like parsers in C
   
      Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.
   
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.
   
   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */


/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     LID = 258,
     ID = 259,
     CONST = 260,
     INCLUDE = 261,
     USING = 262,
     NAMESPACE = 263,
     STD = 264,
     IOSTREAM = 265,
     MATH_H = 266,
     STRING_H = 267,
     CONST_ = 268,
     INT = 269,
     FLOAT = 270,
     DOUBLE = 271,
     VOID = 272,
     STRUCT = 273,
     IF = 274,
     WHILE = 275,
     FOR = 276,
     CIN = 277,
     COUT = 278,
     RETURN = 279,
     LEFT_BRACE = 280,
     RIGHT_BRACE = 281,
     LEFT_PAREN = 282,
     RIGHT_PAREN = 283,
     COMMA = 284,
     SEMICOLON = 285,
     LESS_THAN = 286,
     GREATER_THAN = 287,
     EQUAL = 288,
     NOT_EQUAL = 289,
     EQUAL_EQUAL = 290,
     LESS_THAN_OR_EQUAL = 291,
     GREATER_THAN_OR_EQUAL = 292,
     PLUS = 293,
     MINUS = 294,
     MULTIPLY = 295,
     DIVIDE = 296,
     MODULO = 297,
     LEFT_SHIFT = 298,
     RIGHT_SHIFT = 299,
     HASH = 300,
     INCREMENT = 301,
     DECREMENT = 302
   };
#endif



#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef int YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif

extern YYSTYPE yylval;


