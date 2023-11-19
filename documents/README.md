classDiagram
    class UnknownClass {
        +execute() : void
        +undo() : void
        +redo() : void
    }
    class AssignmentTest {
    }
    class Caretaker {
        +saveTeamToUndo(Team team) : void
        +saveTeamToRedo(Team team) : void
        +savePlayerToUndo(Player player) : void
        +savePlayerToRedo(Player player) : void
        +undo() : void
        +redo() : voids
        +clearRedoList() : void
    }
    class FootballTeam {
        +displayTeam() : void
        +toString() : String
    }
    class UnknownClass {
        +restore() : void
    }
    class MementoPlayer {
        -player : Player
        -name : String
        -position : int
        +restore() : void
    }
    class MementoTeam {
        -team : Team
        -name : String
        +restore() : void
    }
    class Player {
        -playerID : String
        -name : String
        -position : int
        +getPlayerID() : String
        +getPosition() : int
        +setPosition(int position) : void
        +getName() : String
        +setName(String name) : void
        +toString() : String
    }
    class Team {
        -teamID : String
        -name : String
        +getTeamID() : String
        +getName() : String
        +setName(String name) : void
        +addPlayer(Player player) : void
        +addPlayer(Player player; int index) : void
        +removePlayer(Player player) : void
        +updatePlayerPosition(String playerID; int newPosition) : void
    }
    class VolleyballTeam {
        +displayTeam() : void
        +toString() : String
    }
