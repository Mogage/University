package com.socialNetwork.network;

import com.socialNetwork.domain.Friendship;
import com.socialNetwork.domain.User;
import com.socialNetwork.exceptions.NetworkException;

import java.util.*;

public class MainNetwork implements Network{
    private final Map<Long, List<Long>> network = new HashMap<>();

    @Override
    public void add(User entity) {
        network.put(entity.getId(), new ArrayList<>());
    }

    @Override
    public void remove(User entity) {
        Long idToRemove = entity.getId();
        for (List<Long> friends : network.values()) {
            friends.remove(idToRemove);
        }
        network.remove(idToRemove);
    }

    private Boolean areFriends(Long entity1, Long entity2) {
        List<Long> friends = network.get(entity1);
        for (Long friendId : friends) {
            if (Objects.equals(friendId, entity2)) {
                return true;
            }
        }
        return false;
    }

    @Override
    public void makeFriends(Friendship friendship) throws NetworkException {
        if (areFriends(friendship.getIdUser1(), friendship.getIdUser2())) {
            throw new NetworkException("These people are already friends.\n");
        }
        network.get(friendship.getIdUser1()).add(friendship.getIdUser2());
        network.get(friendship.getIdUser2()).add(friendship.getIdUser1());
    }

    @Override
    public void removeFriends(Friendship friendship) throws NetworkException {
        if (!areFriends(friendship.getIdUser1(), friendship.getIdUser2())) {
            throw new NetworkException("These people are not friends.\n");
        }
        network.get(friendship.getIdUser1()).remove(friendship.getIdUser2());
        network.get(friendship.getIdUser2()).remove(friendship.getIdUser1());
    }

    private int dfs(long[] users, Long id, int communityNumber) {
        users[Math.toIntExact(id)] = communityNumber;
        int numberOfPath = 0;
        List<Long> friends = network.get(id);
        for (long idFriend : friends) {
            if (users[Math.toIntExact(idFriend)] == 0) {
                int path = dfs(users, idFriend, communityNumber);
                numberOfPath = Math.max(numberOfPath, path);
            }
        }
        return numberOfPath + 1;
    }

    private Long getMaxId() {
        long max = 0L;
        for (Long id : network.keySet()) {
            max = Math.max(max, id);
        }
        return max;
    }

    private long[] getCommunities() {
        long[] users = new long[(int) (getMaxId() + 1)];
        int communityNumber = 1;
        for (long id : network.keySet()) {
            if (users[Math.toIntExact(id)] == 0) {
                dfs(users, id, communityNumber);
                communityNumber = communityNumber + 1;
            }
        }
        return users;
    }

    private Map<Long, Integer> getPopulation() {
        long[] users = getCommunities();
        Map<Long, Integer> communitiesPopulation = new HashMap<>();

        for (int count = 1; count < users.length; count = count + 1) {
            if (users[count] == 0) {
                continue;
            }

            if (!communitiesPopulation.containsKey(users[count])) {
                communitiesPopulation.put(users[count], 0);
            }
            communitiesPopulation.put(users[count], communitiesPopulation.get(users[count]) + 1);
        }

        return communitiesPopulation;
    }

    @Override
    public int getNumberOfCommunities() {
        Map<Long, Integer> communitiesPopulation = getPopulation();

        return communitiesPopulation.size();
    }

    @Override
    public List<Long> getMostPopulatedCommunity() {
        long[] users = new long[(int) (getMaxId() + 1)];
        List<Long> biggestCommunity = new ArrayList<>();
        int communityNumber = 1;
        int maxPath = 0;
        int currentPath;
        long startNode = -1L;
        for (Long id : network.keySet()) {
            if (users[Math.toIntExact(id)] == 0) {
                currentPath = dfs(users, id, communityNumber);
                communityNumber = communityNumber + 1;
                if (currentPath > maxPath) {
                    maxPath = currentPath;
                    startNode = id;
                }
            }
        }
        Arrays.fill(users, 0L);
        dfs(users, startNode, 1);

        for (int count = 1; count < users.length; count = count + 1) {
            if (users[count] != 1) {
                continue;
            }
            biggestCommunity.add((long) count);
        }

        return biggestCommunity;
    }
}
